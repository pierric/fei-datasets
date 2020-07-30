{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns         #-}
module MXNet.NN.DataIter.Coco(
    module MXNet.NN.DataIter.Common,
    Configuration(..), CocoConfig, conf_width, Coco(..),
    classes, coco,
    cocoImageList, cocoImages, cocoImagesBBoxes, cocoImagesBBoxesMasks,
    loadImage, loadBoundingBoxes, loadMasks,
) where

import           Control.Lens                  (ix, makeLenses, (^?!))
import qualified Data.Aeson                    as Aeson
import           Data.Array.Repa               ((:.) (..), Z (..),
                                                fromListUnboxed)
import qualified Data.Array.Repa               as Repa
import           Data.Conduit
import qualified Data.Conduit.Combinators      as C (yieldMany)
import qualified Data.Conduit.List             as C
import qualified Data.Random                   as RND (StdRandom (..), runRVar,
                                                       shuffleN)
import qualified Data.Store                    as Store
import qualified Data.Vector.Storable          as SV (unsafeCast)
import           GHC.Float                     (double2Float)
import           GHC.Generics                  (Generic)
import qualified Graphics.Image                as HIP
import qualified Graphics.Image.Interface      as HIP
import qualified Graphics.Image.Interface.Repa as HIP
import           RIO
import qualified RIO.ByteString                as SBS
import           RIO.Directory
import           RIO.FilePath
import qualified RIO.Map                       as M
import qualified RIO.NonEmpty                  as RNE
import qualified RIO.Vector.Boxed              as V
import qualified RIO.Vector.Storable           as SV

import           MXNet.Coco.Mask
import           MXNet.Coco.Types
import           MXNet.NN.DataIter.Common

classes :: V.Vector String
classes = V.fromList [
    "__background__",  -- always index 0
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

data Coco = Coco FilePath String Instance (M.Map Int Int)
    deriving Generic
instance Store.Store Coco

data FileNotFound = FileNotFound String String
    deriving Show
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
              => Bool -> ConduitT () Image m ()
cocoImageList shuffle = do
    Coco _ _ inst _ <- view (datasetConfig . conf_coco)
    let all_images = inst ^. images
    all_images_shuffle <- liftIO $
        if shuffle then
            RND.runRVar (RND.shuffleN (length all_images) (V.toList all_images)) RND.StdRandom
        else
            return $ V.toList all_images
    C.yieldMany all_images_shuffle -- .| C.iterM (liftIO . print)


cocoImages :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
           => Bool -> ConduitT () (String, ImageTensor, ImageInfo) m ()
cocoImages shuffle = cocoImageList shuffle .| C.mapM build
    where
        build image = do
            let filename = image ^. img_file_name
            (img, info) <- loadImage image
            return $!! (filename, img, info)


cocoImagesBBoxes :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
                 => Bool -> ConduitT () (String, ImageTensor, ImageInfo, GTBoxes) m ()
cocoImagesBBoxes shuffle = cocoImageList shuffle .| C.mapM build .| C.catMaybes
    where
        build image = do
            let filename = image ^. img_file_name
            (img, info) <- loadImage image
            mboxes <- loadBoundingBoxes image
            case mboxes of
              Nothing    -> return Nothing
              Just boxes -> return $!! Just (filename, img, info, boxes)


cocoImagesBBoxesMasks :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
                      => Bool -> ConduitT () (String, ImageTensor, ImageInfo, GTBoxes, Masks) m ()
cocoImagesBBoxesMasks shuffle = cocoImageList shuffle .| C.mapM build .| C.catMaybes
    where
        build image = do
            let filename = image ^. img_file_name
            (img, info) <- loadImage image
            mboxes <- loadBoundingBoxes image
            mmasks  <- loadMasks image
            case liftA2 (,) mboxes mmasks of
              Nothing             -> return Nothing
              Just (boxes, masks) -> return $!! Just (filename, img, info, boxes, masks)


getImageScale :: Image -> Int -> (Float, Int, Int)
getImageScale img size
  | oriW >= oriH = (sizeF / oriW, size, floor (oriH * sizeF / oriW))
  | otherwise    = (sizeF / oriH, floor (oriW * sizeF / oriH), size)
    where
        oriW = fromIntegral $ img ^. img_width
        oriH = fromIntegral $ img ^. img_height
        sizeF = fromIntegral size


loadImage :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
    => Image -> m (ImageTensor, ImageInfo)
loadImage img = do
    Coco base datasplit _ _ <- view (datasetConfig . conf_coco)
    width <- view (datasetConfig . conf_width)

    let imgFilePath = base </> datasplit </> img ^. img_file_name
    imgRGB <- liftIO $ raiseLeft (FileNotFound imgFilePath) $ HIP.readImage imgFilePath

    let (scale, imgH, imgW) = getImageScale img width
        imgInfo = fromListUnboxed (Z :. 3) [fromIntegral imgH, fromIntegral imgW, scale]

        imgResized = HIP.resize HIP.Bilinear HIP.Edge (imgH, imgW) (imgRGB :: HIP.Image HIP.VS HIP.RGB Double)
        imgPadded  = HIP.canvasSize (HIP.Fill $ HIP.PixelRGB 0.5 0.5 0.5) (width, width) imgResized
        imgRepa    = Repa.fromUnboxed (Z:.width:.width:.3) $
                        SV.convert $
                        SV.unsafeCast $
                        HIP.toVector imgPadded

    imgEval <- transform $ Repa.map double2Float imgRepa
    return (Repa.computeUnboxedS imgEval, imgInfo)


loadBoundingBoxes :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "coco", MonadIO m)
    => Image -> m (Maybe GTBoxes)
loadBoundingBoxes img = do
    Coco _ _ inst cat_to_classid <- view (datasetConfig . conf_coco)
    size <- view (datasetConfig . conf_width)

    let imgAnns = V.filter (\ann -> ann ^. ann_image_id == imageId) (inst ^. annotations)
        (scale, _, _) = getImageScale img size
        gt_boxes = V.fromList $ catMaybes $ map (makeGTBox cat_to_classid scale) $ V.toList imgAnns

    return $ if V.null gt_boxes then Nothing else Just gt_boxes
    where
        imageId = img ^. img_id
        width   = img ^. img_width
        height  = img ^. img_height
        makeGTBox cat_to_classid scale ann =
            let (x0, y0, x1, y1) = cleanBBox (ann ^. ann_bbox)
                classId = cat_to_classid ^?! ix (ann ^. ann_category_id)
            in
            if ann ^. ann_area > 0 && x1 > x0 && y1 > y0
                then Just $ fromListUnboxed (Z :. 5) [x0*scale, y0*scale, x1*scale, y1*scale, fromIntegral classId]
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
        (_, imgH, imgW) = getImageScale img size
    masks <- V.mapM (getMask imgH imgW size) imgAnns
    return $ if V.null masks then Nothing else Just masks
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
            let -- always single channel, since we have merged the masks
                -- also note that HIP uses image HxW
                image      = HIP.fromRepaArrayS $
                                Repa.transpose $
                                Repa.map (HIP.PixelY . (*255)) $
                                Repa.reshape (Z :. height :. width) mask
                imgResized = HIP.resize HIP.Bilinear HIP.Edge
                                (upH, upW)
                                image
                imgPadded  = HIP.canvasSize (HIP.Fill $ HIP.PixelY 0)
                                (size, size)
                                imgResized
            return $ Repa.computeUnboxedS $ Repa.map (\(HIP.PixelY e) -> e) $ HIP.toRepaArray imgPadded
