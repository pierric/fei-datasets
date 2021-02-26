{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns         #-}
module MXNet.NN.DataIter.Coco(
    module MXNet.NN.DataIter.Common,
    Configuration(..), CocoConfig, conf_width, Coco(..),
    classes, coco,
    cocoImageList, cocoImages, cocoImagesBBoxes, cocoImagesBBoxesMasks,
    loadImage, loadBoundingBoxes, loadMasks,
    augmentWithBBoxes,
) where

import           Control.Lens                (ix, makeLenses, (^?!))
import qualified Data.Aeson                  as Aeson
import           Data.Conduit
import qualified Data.Conduit.Combinators    as C (yieldMany)
import qualified Data.Conduit.List           as C
import qualified Data.Random                 as RND (runRVar, shuffleN,
                                                     stdUniform)
import           Data.Random.Source.StdGen   (StdGen)
import qualified Data.Store                  as Store
import qualified Data.Vector.Storable        as SV (unsafeCast)
import           GHC.Float                   (double2Float)
import           GHC.Generics                (Generic)
import qualified Graphics.Image              as HIP
import qualified Graphics.Image.Interface    as HIP
import           RIO
import qualified RIO.ByteString              as SBS
import           RIO.Directory
import           RIO.FilePath
import qualified RIO.Map                     as M
import qualified RIO.NonEmpty                as RNE
import qualified RIO.Vector.Boxed            as V
import qualified RIO.Vector.Storable         as SV

import           Fei.Einops
import           MXNet.Base
import           MXNet.Base.Operators.Tensor (_Pad, __image_resize, _reverse)
import           MXNet.Base.Tensor           (cast, mulScalar, reshape,
                                              rsubScalar, splitBySections,
                                              stack)
import           MXNet.Coco.Mask
import           MXNet.Coco.Types
import           MXNet.NN.DataIter.Common

classes :: V.Vector String
classes = V.fromList [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"]

data Coco
  = Coco FilePath String Instance (M.Map Int Int)
  deriving (Generic)
instance Store.Store Coco

data FileNotFound
  = FileNotFound String String
  deriving (Show)
instance Exception FileNotFound

data instance Configuration "coco" = CocoConfig {
    _conf_coco :: Coco,
    _conf_width :: Int,
    _conf_mean :: (Float, Float, Float),
    _conf_std :: (Float, Float, Float)
}

makeLenses 'CocoConfig
type CocoConfig = Configuration "coco"

instance HasDatasetConfig CocoConfig where
    type DatasetTag CocoConfig = "coco"
    datasetConfig = id

instance ImageDataset "coco" where
    imagesMean = conf_mean
    imagesStdDev = conf_std

cached :: Store.Store a => String -> IO a -> IO a
cached name action = do
    createDirectoryIfMissing True "cache"
    hitCache <- doesFileExist path
    if hitCache then
        SBS.readFile path >>= Store.decodeIO
    else do
        obj <- action
        SBS.writeFile path (Store.encode obj)
        return obj
  where
    path = "cache/" ++ name

coco :: String -> String -> IO Coco
coco base datasplit = cached (datasplit ++ ".store") $ do
    let annotationFile = base </> "annotations" </> ("instances_" ++ datasplit ++ ".json")
    inst <- raiseLeft (FileNotFound annotationFile) $ Aeson.eitherDecodeFileStrict' annotationFile
    let cat_to_classid = M.fromList $ V.toList $ V.map
           (\cat -> (cat ^. odc_id, get_cat_classid (cat ^. odc_name)))
           (inst ^. categories)
    return $ Coco base datasplit inst cat_to_classid

    where
        get_cat_classid (flip V.elemIndex classes -> Just index) = index
        get_cat_classid _ = error "index not found in classes"


cocoImageList :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
              => IORef StdGen -> ConduitT () Image m ()
cocoImageList rand_gen = do
    Coco _ _ inst _ <- view (datasetConfig . conf_coco)
    let all_images = inst ^. images
    all_images_shuffle <- liftIO $
        RND.runRVar (RND.shuffleN (length all_images) (V.toList all_images)) rand_gen
    C.yieldMany all_images_shuffle -- .| C.iterM (liftIO . print)


cocoImages :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
           => IORef StdGen -> ConduitT () (String, ImageTensor, ImageInfo) m ()
cocoImages rand_gen = cocoImageList rand_gen .| C.mapM build
    where
        build image = do
            let filename = image ^. img_file_name
            (img, info) <- loadImage image
            return $!! (filename, img, info)


cocoImagesBBoxes :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
                 => IORef StdGen -> ConduitT () (String, ImageTensor, ImageInfo, GTBoxes) m ()
cocoImagesBBoxes rand_gen = cocoImageList rand_gen .| C.mapM build .| C.catMaybes
    where
        build image = do
            let filename = image ^. img_file_name
            (img, info) <- loadImage image
            mboxes <- loadBoundingBoxes image
            case mboxes of
              Nothing    -> return Nothing
              Just boxes -> return $!! Just (filename, img, info, boxes)


cocoImagesBBoxesMasks :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
                      => IORef StdGen -> ConduitT () (String, ImageTensor, ImageInfo, GTBoxes, Masks) m ()
cocoImagesBBoxesMasks rand_gen = cocoImageList rand_gen .| C.mapM build .| C.catMaybes
    where
        build image = do
            let filename = image ^. img_file_name
            (img, info) <- loadImage image
            mboxes <- loadBoundingBoxes image
            mmasks  <- loadMasks image
            case liftA2 (,) mboxes mmasks of
              Nothing             -> return Nothing
              Just (boxes, masks) -> return $!! Just (filename, img, info, boxes, masks)


augmentWithBBoxes :: MonadIO m
                  => IORef StdGen
                  -> (String, ImageTensor, ImageInfo, GTBoxes)
                  -> m (String, ImageTensor, ImageInfo, GTBoxes)
augmentWithBBoxes rand_gen inp@(ident, img, info, bboxes) = liftIO $ do
    do_flip <- RND.runRVar RND.stdUniform rand_gen
    if not do_flip
    then return inp
    else do
        img_flipped <- prim _reverse (#data := img .& #axis := [2] .& Nil)
        [c, h, w] <- ndshape img
        [x0s, y0s, x1s, y1s, scs] <- splitBySections 5 1 True bboxes
        x0s' <- rsubScalar (fromIntegral w - 1) x1s
        x1s' <- rsubScalar (fromIntegral w - 1) x0s
        boxes_flipped <- stack (-1) [x0s', y0s, x1s', y1s, scs]
        return (ident, img_flipped, info, boxes_flipped)


loadImage :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
    => Image -> m (ImageTensor, ImageInfo)
loadImage img = do
    Coco base datasplit _ _ <- view (datasetConfig . conf_coco)
    width <- view (datasetConfig . conf_width)

    let imgFilePath = base </> datasplit </> img ^. img_file_name
    imgRGB <- liftIO $ raiseLeft (FileNotFound imgFilePath) $ HIP.readImage imgFilePath

    let (scale, imgH, imgW) = getImageScale (img ^. img_height) (img ^. img_width) width
        imgResized = HIP.resize HIP.Bilinear HIP.Edge (imgH, imgW) (imgRGB :: HIP.Image HIP.VS HIP.RGB Double)
        imgPadded  = HIP.canvasSize (HIP.Fill $ HIP.PixelRGB 0.5 0.5 0.5) (width, width) imgResized

    info <- liftIO $ fromVector [3] [fromIntegral imgH, fromIntegral imgW, scale]
    let img_padded_vec = SV.unsafeCast $ HIP.toVector imgPadded :: SV.Vector Double
    img_f <- liftIO $ fromVector [width, width, 3] img_padded_vec >>= cast #float32
    img_f <- transform img_f
    return (img_f, info)


loadBoundingBoxes :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
    => Image -> m (Maybe GTBoxes)
loadBoundingBoxes img = do
    Coco _ _ inst cat_to_classid <- view (datasetConfig . conf_coco)
    size <- view (datasetConfig . conf_width)

    let imgAnns = V.filter (\ann -> ann ^. ann_image_id == imageId) (inst ^. annotations)
        (scale, _, _) = getImageScale (img ^. img_height) (img ^. img_width) size
        gt_boxes = catMaybes $ map (makeGTBox cat_to_classid scale) $ V.toList imgAnns

    if null gt_boxes
    then return Nothing
    else liftIO $ do
        gts <- mapM (fromVector [5]) gt_boxes
        gts <- stack 0 gts
        return $!! Just gts

    where
        imageId = img ^. img_id
        width   = img ^. img_width
        height  = img ^. img_height
        makeGTBox cat_to_classid scale ann =
            let (x0, y0, x1, y1) = cleanBBox (ann ^. ann_bbox)
                classId = cat_to_classid ^?! ix (ann ^. ann_category_id)
            in
            if ann ^. ann_area > 0 && x1 > x0 && y1 > y0
                then Just [x0*scale, y0*scale, x1*scale, y1*scale, fromIntegral classId]
                else Nothing
        cleanBBox (x, y, w, h) =
            let x0 = max 0 x
                y0 = max 0 y
                x1 = min (fromIntegral width - 1)  (x0 + max 0 (w-1))
                y1 = min (fromIntegral height - 1) (y0 + max 0 (h-1))
             in (x0, y0, x1, y1)


loadMasks :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
          => Image -> m (Maybe Masks)
loadMasks img = do
    Coco _ _ inst _ <- view (datasetConfig . conf_coco)
    size <- view (datasetConfig . conf_width)

    let imgAnns = V.filter (\ann -> ann ^. ann_image_id == imageId) (inst ^. annotations)
        (_, imgH, imgW) = getImageScale (img ^. img_height) (img ^. img_width) size
    masks <- V.mapM (getMask imgH imgW size) imgAnns
    if V.null masks
    then return Nothing
    else liftIO $ do
        masks <- stack 0 (V.toList masks)
        return $!! Just masks

    where
        imageId = img ^. img_id
        width   = img ^. img_width
        height  = img ^. img_height
        getMask upH upW size anno = liftIO $ do
            crle <- case anno ^. ann_segmentation of
                      SegRLE cnts _    -> frUncompressedRLE cnts height width
                      SegPolygon (RNE.nonEmpty -> Just polys) ->
                          frPoly (RNE.map SV.fromList polys) height width
                      _ -> throwString "Cannot build CRLE"
            crle <- merge crle False
            mask <- decode crle
            mask <- rearrange mask "c w h -> h w c" []
            mask <- prim __image_resize
                      (#data := mask .& #size := [upW, upH] .& Nil)
            -- uint8 -> float32, value stay in [0, 1]
            mask <- cast #float32 mask
            -- pad only works for 4/5d ndarray
            mask <- rearrange mask "(a b w) h c -> a b w h c" [#a .== 1, #b .== 1]
            mask <- prim _Pad
                      (#data := mask .& #mode := #constant .& #constant_value := 0
                    .& #pad_width := [0, 0, 0, 0, 0, size - upH, 0, size - upW, 0, 0] .& Nil)
            reshape [size, size] mask

