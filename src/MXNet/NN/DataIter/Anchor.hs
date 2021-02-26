{-# LANGUAGE TemplateHaskell #-}
module MXNet.NN.DataIter.Anchor where

import           Control.Lens                 (ix, makeLenses, (^?!))
import           Data.Random                  (StdRandom (..), runRVar,
                                               shuffleNofM)
import qualified Data.Vector                  as V (fold1M)
import qualified Data.Vector.Storable.Mutable as SVM
import           RIO
import qualified RIO.HashMap                  as M
import qualified RIO.NonEmpty                 as NE
import qualified RIO.Vector.Storable          as SV

import           Fei.Einops
import           MXNet.Base
import           MXNet.Base.Operators.Tensor  (__contrib_box_encode,
                                               __contrib_box_iou, __set_value,
                                               _slice)
import           MXNet.Base.ParserUtils       (decimal, list, parseR, rational,
                                               tuple)
import           MXNet.NN.DataIter.Common     (Anchors, GTBoxes, getImageScale)
import           MXNet.NN.Layer               (addScalar, add_, and_, argmax,
                                               broadcastLike, copy, eqBroadcast,
                                               expandDims, geqScalar, gtScalar,
                                               leqScalar, ltScalar, max_,
                                               mulBroadcast, mul_, or_, reshape,
                                               rsubScalar, sliceAxis,
                                               splitBySections, squeeze, stack,
                                               subScalar, sum_)

data AnchorError = BadDimension deriving (Show)
instance Exception AnchorError

data Configuration
  = Configuration
      { _conf_anchor_scales    :: [Int]
      , _conf_anchor_ratios    :: [Float]
      , _conf_anchor_base_size :: Int
      , _conf_allowed_border   :: Int
      , _conf_fg_num           :: Int
      , _conf_batch_num        :: Int
      , _conf_bg_overlap       :: Float
      , _conf_fg_overlap       :: Float
      }
  deriving (Show)
makeLenses ''Configuration

anchors :: (Int, Int) -> Int -> Int -> [Int] -> [Float] -> IO Anchors
anchors (height, width) stride base_size scales ratios = do
    base <- baseAnchors base_size scales ratios
    variations <- sequence [ make anch offX offY
                           | offY <- grid height
                           , offX <- grid width
                           , anch <- base ]
    stack 0 variations
  where
    grid size = map fromIntegral [0, stride .. size * stride-1]
    make anch offX offY = do
        offs <- fromVector [4] [offX, offY, offX, offY]
        add_ anch offs

baseAnchors :: Int -> [Int] -> [Float] -> IO [NDArray Float]
baseAnchors base_size scales ratios = sequence [makeBase s r | r <- ratios, s <- scales]
  where
    makeBase scale ratio =
        let sizeF = fromIntegral base_size - 1
            (w, h, x, y) = whctr (0, 0, sizeF, sizeF)
            ws = round $ sqrt (w * h / ratio) :: Int
            hs = round $ (fromIntegral ws) * ratio :: Int
        in mkanchor x y (fromIntegral $ ws * scale) (fromIntegral $ hs * scale)

whctr :: (Float, Float, Float, Float) -> (Float, Float, Float, Float)
whctr (x0, y0, x1, y1) = (w, h, x, y)
  where
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    x = x0 + 0.5 * (w - 1)
    y = y0 + 0.5 * (h - 1)

mkanchor :: Float -> Float -> Float -> Float -> IO (NDArray Float)
mkanchor x y w h = fromVector [4] [x - hW, y - hH, x + hW, y + hH]
  where
    hW = 0.5 * (w - 1)
    hH = 0.5 * (h - 1)

overlapMatrix :: HasCallStack => NDArray Float -> GTBoxes -> Anchors -> IO (NDArray Float)
overlapMatrix mask gtBoxes anBoxes = do
    -- iou is of the shape [numGTs, numAnchors]
    iou <- prim __contrib_box_iou (#lhs := gtBoxes .& #rhs := anBoxes .& #format := #corner .& Nil)
    -- rearrange mask from [numAnchors] -> [1, numAnchors]
    mask <- rearrange mask "(n a) -> n a" [#n .== 1]
    mulBroadcast iou mask

type Labels  = NDArray Float -- DIM2
type Targets = NDArray Float -- DIM2
type Weights = NDArray Float -- DIM2

assign :: (HasCallStack, MonadReader Configuration m, MonadIO m) =>
    GTBoxes -> Int -> Int -> Anchors -> m (Labels, Targets, Weights)
assign gtBoxes imWidth imHeight anBoxes = do
    -- #GT should never be 0
    --    goodIndices <- filterGoodIndices anBoxes
    --    liftIO $ do
    --        indices <- runRVar (shuffleN (Set.size goodIndices) (Set.toList goodIndices)) StdRandom
    --        labels  <- SVM.replicate numAnchors (-1)
    --        forM_ indices $
    --            flip (SVM.write labels) 0
    --        labels  <- fromVector [numAnchors, 1] $ SV.unsafeFreeze labels
    --        targets <- zeros [numAnchors, 4]
    --        weights <- zeros [numAnchors, 4]
    --        return (labels, targets, weights)

    _fg_overlap <- view conf_fg_overlap
    _bg_overlap <- view conf_bg_overlap
    _batch_num  <- view conf_batch_num
    _fg_num     <- view conf_fg_num
    _allowed_border <- fromIntegral <$> view conf_allowed_border

    liftIO $ do
        gtBoxes <- sliceAxis gtBoxes 1 0 (Just 4)
        anchor_valid <- filterGoodIndices anBoxes _allowed_border
        overlaps <- overlapMatrix anchor_valid gtBoxes anBoxes

        -- for each GT, the hightest overlapping anchors are FG.
        fgs1     <- max_ overlaps (Just [1]) True >>= eqBroadcast overlaps
        fgs1     <- sum_ fgs1 (Just [0]) False >>= gtScalar 0

        -- FG anchors that have overlapping with any GT >= thresh
        max_iou_per_anchor <- max_ overlaps (Just [0]) False
        fgs2 <- geqScalar _fg_overlap max_iou_per_anchor

        fgs  <- or_ fgs1 fgs2
        -- subsample FG anchors if there are too many
        (numFG, fgs) <- atMost _fg_num fgs

        -- BG anchors that have overlapping with all GT < thresh
        bgs  <- leqScalar _bg_overlap max_iou_per_anchor
        (numBG, bgs) <- atMost (_batch_num - min numFG _fg_num) bgs

        -- set fg to 2, bg to 1, and invalid to 0
        labels <- rsubScalar 1 bgs >>= mul_ fgs >>= addScalar 1 >>= mul_ anchor_valid
        -- set fg to 1, bg to 0, and invalid to -1
        labels <- subScalar 1 labels

        matches <- argmax overlaps (Just 0) False
        means   <- zeros [4]
        stds    <- ones  [4]

        -- __contrib_box_encode expects (B, N, ..)
        gtBoxesB <- expandDims 0 gtBoxes
        anBoxesB <- expandDims 0 anBoxes
        matchesB <- expandDims 0 matches
        labelsB  <- expandDims 0 labels

        [targetsB, weightsB] <- primMulti __contrib_box_encode
                                    (#refs    := gtBoxesB
                                  .& #anchors := anBoxesB
                                  .& #matches := matchesB
                                  .& #samples := labelsB
                                  .& #means   := means
                                  .& #stds    := stds .& Nil)
        -- targets: (N, 4)
        targets <- squeeze (Just [0]) targetsB
        -- weights: (N, 4)
        weights <- squeeze (Just [0]) weightsB
        -- labels:  (N, 1)
        labels <- expandDims 1 labels

        return (labels, targets, weights)
  where
    filterGoodIndices :: Anchors -> Float -> IO (NDArray Float)
    filterGoodIndices anBoxes _allowed_border = do
        [x0, y0, x1, y1] <- splitBySections 4 1 True anBoxes
        flag1 <- geqScalar (-_allowed_border) x0
        flag2 <- geqScalar (-_allowed_border) y0
        flag3 <- ltScalar (fromIntegral imWidth  + _allowed_border) x1
        flag4 <- ltScalar (fromIntegral imHeight + _allowed_border) y1
        V.fold1M and_ [flag1, flag2, flag3, flag4]

    -- find 1's, and keep at most `max` number of those in a 1D array.
    atMost :: Int -> NDArray Float -> IO (Int, NDArray Float)
    atMost max array = do
        vec <- toVector array
        let ind = SV.elemIndices 1 vec
            num = SV.length ind
        dis <- if num > max
               then flip runRVar StdRandom $ shuffleNofM (num - max) num (SV.toList ind)
               else pure []
        ret <- fromVector [SV.length vec] $
               let upd :: PrimMonad m => SV.MVector (PrimState m) Float -> m ()
                   upd v = forM_ dis (\i -> SVM.write v i 0)
                in SV.modify upd vec
        return (num - length dis, ret)

--
-- Symbol for Anchor Generator
--
data AnchorGeneratorProp
  = AnchorGeneratorProp
      { _ag_ratios        :: [Float]
      , _ag_scales        :: [Int]
      , _ag_anchors_alloc :: NDArray Float
      }
makeLenses ''AnchorGeneratorProp

instance CustomOperationProp AnchorGeneratorProp where
    prop_list_arguments _        = ["feature"]
    prop_list_outputs _          = ["anchors"]
    prop_list_auxiliary_states _ = []
    prop_infer_shape prop [feature_shape] =
        let STensor [_, _, h, w] = feature_shape
            num_scales = length (prop ^. ag_scales)
            num_ratios = length (prop ^. ag_ratios)
            num_anchs  = num_scales * num_ratios * h * w
            anchors_shape        = STensor [1, num_anchs, 4]
        in ([feature_shape], [anchors_shape], [])
    prop_declare_backward_dependency _ _ _ _ = []

    data Operation AnchorGeneratorProp = AnchorGenerator AnchorGeneratorProp
    prop_create_operator prop _ _ = return (AnchorGenerator prop)

instance CustomOperation (Operation AnchorGeneratorProp) where
    forward (AnchorGenerator prop) [ReqWrite] [feature] [anchors] _ _ = do
        -- :param: feature, shape of (1, C, H, W)
        -- :param: anchors, shape of (1, N, 4), where N is number of anchors

        let alloc = prop ^. ag_anchors_alloc

        -- get the height, width of the feature (B,C,H,W)
        [_,_,h,w] <- NE.toList <$> ndshape (NDArray feature :: NDArray Float)
        let beg = [0,0,0,0]
            end = [1,1,h,w]
        ret <- prim _slice (#data := alloc .& #begin:= beg .& #end:= end .& Nil)
        ret <- reshape [1,-1,4] ret
        void $ copy ret (NDArray anchors)

    backward _ [ReqWrite] _ _ [in_grad_0] _ _ = do
        -- type annotation is necessary, because only a general form
        -- can be inferred.
        let set_zeros = __set_value (#src := 0 .& Nil) :: TensorApply NDArrayHandle
        void $ set_zeros (Just ([in_grad_0]))

buildAnchorGenerator :: HasCallStack => [(Text, Text)] -> IO AnchorGeneratorProp
buildAnchorGenerator params = do
    allocV <- anchors alloc_size stride base_size scales ratios

    let (height, width) = alloc_size
        num_scales = length scales
        num_ratios = length ratios

    allocA <- rearrange allocV "(a b h w c) d -> a b h w (c d)"
                [ #a .== 1, #b .== 1
                , #h .== height, #w .== width
                , #c .== num_scales * num_ratios
                , #d .== 4]

    return $ AnchorGeneratorProp
        { _ag_scales = scales
        , _ag_ratios = ratios
        , _ag_anchors_alloc = allocA
        }
  where
    paramsM    = M.fromList params
    stride     = parseR decimal         $ paramsM ^?! ix "stride"
    scales     = parseR (list decimal)  $ paramsM ^?! ix "scales"
    ratios     = parseR (list rational) $ paramsM ^?! ix "ratios"
    base_size  = parseR decimal         $ paramsM ^?! ix "base_size"
    alloc_size = parseR (tuple decimal) $ paramsM ^?! ix "alloc_size"
