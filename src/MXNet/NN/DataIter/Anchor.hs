{-# LANGUAGE TemplateHaskell #-}
module MXNet.NN.DataIter.Anchor where

import qualified Data.IntSet as Set
import Control.Exception
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as UVM
import Control.Lens (view, makeLenses)
import Control.Monad.Reader
import Data.Random (shuffleN, runRVar, StdRandom(..))
import Data.Array.Repa (Array, DIM1, DIM2, D, U, (:.)(..), Z (..), All(..), (+^), fromListUnboxed)
import qualified Data.Array.Repa as Repa

data AnchorError = BadDimension
  deriving Show
instance Exception AnchorError

type Anchor r = Array r DIM1 Float
type GTBox r = Array r DIM1 Float

data Configuration = Configuration {
    _conf_anchor_scales :: [Int],
    _conf_anchor_ratios :: [Float],
    _conf_allowed_border :: Int,
    _conf_fg_num :: Int,
    _conf_batch_num :: Int,
    _conf_bg_overlap :: Float,
    _conf_fg_overlap :: Float
} deriving Show
makeLenses ''Configuration

anchors :: MonadReader Configuration m =>
    Int -> Int -> Int -> m (V.Vector (Anchor U))
anchors stride width height = do
    scales <- view conf_anchor_scales
    ratios <- view conf_anchor_ratios
    base   <- baseAnchors stride
    return $ V.fromList
        [ Repa.computeS $ anch +^ offs
        | offY <- grid height
        , offX <- grid width
        , anch <- base
        , let offs = fromListUnboxed (Z :. 4) [offX, offY, offX, offY]]
  where
    grid size = map fromIntegral [0, stride .. size * stride-1]

baseAnchors :: MonadReader Configuration m =>
    Int -> m ([Anchor U])
baseAnchors size = do
    scales <- view conf_anchor_scales
    ratios <- view conf_anchor_ratios
    return [makeBase s r | r <- ratios, s <- scales]
  where
    makeBase :: Int -> Float -> Anchor U
    makeBase scale ratio =
        let sizeF = fromIntegral size - 1
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

mkanchor :: Float -> Float -> Float -> Float -> Anchor U
mkanchor x y w h = fromListUnboxed (Z :. 4) [x - hW, y - hH, x + hW, y + hH]
  where
    hW = 0.5 * (w - 1)
    hH = 0.5 * (h - 1)

(#!) :: Array U DIM1 Float -> Int -> Float
(#!) = Repa.unsafeLinearIndex

(%!) :: V.Vector a -> Int -> a
(%!) = (V.!)

overlapMatrix :: Set.IntSet -> V.Vector (GTBox U) -> V.Vector (Anchor U) -> Array D DIM2 Float
overlapMatrix goodIndices gtBoxes anBoxes = Repa.fromFunction (Z :. width :. height) calcOvp
  where
    width = V.length gtBoxes
    height = V.length anBoxes

    calcArea box = (box #! 2 - box #! 0 + 1) * (box #! 3 - box #! 1 + 1)
    areaA = V.map calcArea anBoxes
    areaG = V.map calcArea gtBoxes

    calcOvp (Z :. ig :. ia) =
        let gt = gtBoxes %! ig
            anchor = anBoxes %! ia
            iw = min (gt #! 2) (anchor #! 2) - max (gt #! 0) (anchor #! 0)
            ih = min (gt #! 3) (anchor #! 3) - max (gt #! 1) (anchor #! 1)
            areaI = iw * ih
            areaU = areaA %! ia + areaG %! ig - areaI
        in if Set.member ia goodIndices && iw > 0 && ih > 0 then areaI / areaU else 0

type Labels  = Repa.Array U DIM1 Float -- UV.Vector Int
type Targets = Repa.Array U DIM2 Float -- UV.Vector (Float, Float, Float, Float)
type Weights = Repa.Array U DIM2 Float -- UV.Vector (Float, Float, Float, Float)

assign :: (MonadReader Configuration m, MonadIO m) =>
    V.Vector (GTBox U) -> Int -> Int -> V.Vector (Anchor U) -> m (Labels, Targets, Weights)
assign gtBoxes imWidth imHeight anBoxes
    | numGT == 0 = do
        goodIndices <- filterGoodIndices
        liftIO $ do
            indices <- runRVar (shuffleN (Set.size goodIndices) (Set.toList goodIndices)) StdRandom
            labels <- UVM.replicate numLabels (-1)
            forM_ indices $ flip (UVM.write labels) 0
            let targets = UV.replicate (numLabels * 4) 0
                weights = UV.replicate (numLabels * 4) 0
            labels <- UV.unsafeFreeze labels
            let labelsRepa  = Repa.fromUnboxed (Z:.numLabels) labels
                targetsRepa = Repa.fromUnboxed (Z:.numLabels:.4) targets
                weightsRepa = Repa.fromUnboxed (Z:.numLabels:.4) weights
            return (labelsRepa, targetsRepa, weightsRepa)

    | otherwise = do
        _fg_overlap <- view conf_fg_overlap
        _bg_overlap <- view conf_bg_overlap
        _batch_num  <- view conf_batch_num
        _fg_num     <- view conf_fg_num

        goodIndices <- filterGoodIndices

        -- traceShowM ("#Good Anchors:", V.length goodIndices)

        liftIO $ do
            -- TODO filter valid anchor boxes
            -- TODO case when gtBoxes is empty.
            labels <- UVM.replicate numLabels (-1)

            overlaps <- return $ Repa.computeUnboxedS $ overlapMatrix goodIndices gtBoxes anBoxes
            -- for each GT, the hightest overlapping anchor is FG.
            forM_ [0..numGT-1] $ \i -> do
                -- let j = UV.maxIndex $ Repa.toUnboxed $ Repa.computeS $ Repa.slice overlaps (Z :. i :. All)
                let s = Repa.computeUnboxedS $ slice overlaps 0 i
                    m = s #! argMax s
                UV.mapM_ (flip (UVM.write labels) 1) $ UV.findIndices (==m) (Repa.toUnboxed s)

            -- FG anchors that have overlapping with any GT >= thresh
            -- BG anchors that have overlapping with all GT < thresh
            UV.forM_ (UV.indexed $ Repa.toUnboxed $ Repa.foldS max 0 $ Repa.transpose overlaps) $ \(i, m) -> do
                when (Set.member i goodIndices) $ do
                    when (m >= _fg_overlap) $ do
                        -- traceShowM ("FG enable ", m, i)
                        (UVM.write labels i 1)
                    when (m < _bg_overlap) $ do
                        -- s <- UVM.read labels i
                        -- when (s == 1) $ traceShowM ("FG disable ", m, i)
                        (UVM.write labels i 0)

            -- subsample FG anchors if there are too many
            fgs <- UV.findIndices (==1) <$> UV.unsafeFreeze labels
            let numFG = UV.length fgs
            when (numFG > _fg_num) $ do
                indices <- runRVar (shuffleN numFG $ UV.toList fgs) StdRandom
                -- traceShowM ("Disable A", take (numFG - _fg_num) indices)
                forM_ (take (numFG - _fg_num) indices) $
                    flip (UVM.write labels) (-1)

            -- subsample BG anchors if there are too many
            bgs <- UV.findIndices (==0) <$> UV.unsafeFreeze labels
            let numBG = UV.length bgs
                maxBG = _batch_num - min numFG _fg_num
            when (numBG > maxBG) $ do
                indices <- runRVar (shuffleN numBG $ UV.toList bgs) StdRandom
                -- traceShowM ("Disable B", take (numBG - maxBG) indices)
                forM_ (take (numBG - maxBG) indices) $
                    flip (UVM.write labels) (-1)

            -- compute the regression from each FG anchor to its gt
            -- let gts = UV.map (\i -> UV.maxIndex $ Repa.toUnboxed $ Repa.computeS $ Repa.slice overlaps (Z :. i :. All)) fgs
            let gts = UV.map (argMax . Repa.computeUnboxedS . slice overlaps 1) fgs
                gtDiffs = UV.zipWith makeTarget fgs gts
            targets <- UVM.replicate numLabels (0, 0, 0, 0)
            UV.zipWithM_ (UVM.write targets) fgs gtDiffs

            -- indicates which anchors have a regression
            weights <- UVM.replicate numLabels (0, 0, 0, 0)
            UV.forM_ fgs $ flip (UVM.write weights) (1, 1, 1, 1)

            labels  <- UV.unsafeFreeze labels
            targets <- UV.unsafeFreeze targets
            weights <- UV.unsafeFreeze weights
            let labelsRepa  = Repa.fromUnboxed (Z:.numLabels) labels
                targetsRepa = Repa.fromUnboxed (Z:.numLabels:.4) (flattenT targets)
                weightsRepa = Repa.fromUnboxed (Z:.numLabels:.4) (flattenT weights)
            return (labelsRepa, targetsRepa, weightsRepa)
  where
    numGT = V.length gtBoxes
    numLabels = V.length anBoxes

    slice mat 0 ind = Repa.slice mat $ Z :. ind :. All
    slice mat 1 ind = Repa.slice mat $ Z :. All :. ind
    slice _ _ _ = throw BadDimension

    argMax :: Array U DIM1 Float -> Int
    argMax = UV.maxIndex . Repa.toUnboxed

    asTuple :: Array U DIM1 Float -> (Float, Float, Float, Float)
    asTuple box = (box #! 0, box #! 1, box #! 2, box #! 3)

    filterGoodIndices :: MonadReader Configuration m => m Set.IntSet
    filterGoodIndices = do
        _allowed_border <- fromIntegral <$> view conf_allowed_border
        let goodAnchor (x0, y0, x1, y1) =
                x0 >= -_allowed_border &&
                y0 >= -_allowed_border &&
                x1 < fromIntegral imWidth + _allowed_border &&
                y1 < fromIntegral imHeight + _allowed_border
        return $ Set.fromList $ V.toList $ V.findIndices (goodAnchor . asTuple) anBoxes

    makeTarget :: Int -> Int -> (Float, Float, Float, Float)
    makeTarget fgi gti =
        let fgBox = anBoxes %! fgi
            gtBox = gtBoxes %! gti
            (w1, h1, cx1, cy1) = whctr $ asTuple fgBox
            (w2, h2, cx2, cy2) = whctr $ asTuple gtBox
            dx = (cx2 - cx1) / (w1 + 1e-14)
            dy = (cy2 - cy1) / (h1 + 1e-14)
            dw = log (w2 / w1)
            dh = log (h2 / h1)
        in (dx, dy, dw, dh)

    flattenT :: UV.Vector (Float, Float, Float, Float) -> UV.Vector Float
    flattenT = UV.concatMap (\(a,b,c,d) -> UV.fromList [a,b,c,d])

