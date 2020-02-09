{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.NN.DataIter.Coco(
    module MXNet.NN.DataIter.Common,
    Configuration(..), CocoConfig, conf_width,
    classes, coco, cocoImages, cocoImagesAndBBoxes, loadImageAndBBoxes
) where

import Data.Maybe (catMaybes)
import System.FilePath
import System.Directory
import GHC.Generics (Generic)
import GHC.Float (double2Float)
import qualified Data.ByteString as SBS
import qualified Data.Store as Store
import Control.Exception
import Data.Array.Repa (Array, DIM1, DIM3, D, U, (:.)(..), Z (..), Any(..),
    fromListUnboxed, extent, backpermute, extend, (-^), (+^), (*^), (/^))
import qualified Data.Array.Repa as Repa
import Data.Array.Repa.Repr.Unboxed (Unbox)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Graphics.Image as HIP
import qualified Graphics.Image.Interface as HIP
import qualified Data.Aeson as Aeson
import Control.Lens ((^.), view, makeLenses)
import Data.Conduit
import qualified Data.Conduit.Combinators as C (yieldMany)
import qualified Data.Conduit.List as C
import Control.Monad.Reader
import qualified Data.IntMap.Strict as M
import Data.Maybe (fromJust)
import qualified Data.Random as RND (shuffleN, runRVar, StdRandom(..))
import Data.Conduit.ConcurrentMap (concurrentMapM_numCaps)
import Control.Monad.Trans.Resource
import Control.DeepSeq

import MXNet.Coco.Types
import MXNet.NN.DataIter.Common

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

data Coco = Coco FilePath String Instance
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
    inst <- raiseLeft (FileNotFound annotationFile) <$> Aeson.eitherDecodeFileStrict' annotationFile
    return $ Coco base datasplit inst


cocoImages :: (MonadReader CocoConfig m, MonadIO m) => Bool -> ConduitT () Image m ()
cocoImages shuffle = do
    Coco _ _ inst <- view conf_coco
    let all_images = inst ^. images
    all_images <- if shuffle then
                    liftIO $ RND.runRVar (RND.shuffleN (length all_images) (V.toList all_images)) RND.StdRandom
                  else
                    return $ V.toList all_images
    C.yieldMany all_images -- .| C.iterM (liftIO . print)


cocoImagesAndBBoxes :: Bool -> ConduitT () (String, ImageTensor, ImageInfo, GTBoxes) (ReaderT CocoConfig (ResourceT IO)) ()
cocoImagesAndBBoxes shuffle =
    cocoImages shuffle .|
    concurrentMapM_numCaps 16 loadImageAndBBoxes .|
    C.catMaybes


loadImageAndBBoxes :: (MonadReader CocoConfig m, MonadIO m) => Image -> m (Maybe (String, ImageTensor, ImageInfo, GTBoxes))
loadImageAndBBoxes img = do
    Coco base datasplit inst <- view conf_coco
    -- map each category from id to its index in the classes.
    let catTabl = M.fromList $ V.toList $ V.map (\cat -> (cat ^. odc_id, fromJust $ V.elemIndex (cat ^. odc_name) classes)) (inst ^. categories)
        -- get all the bbox and gt for the image
        get_gt_boxes scale img = V.fromList $ catMaybes $ map (makeGTBox img scale) $ V.toList imgAnns
          where
            imageId = img ^. img_id
            imgAnns = V.filter (\ann -> ann ^. ann_image_id == imageId) (inst ^. annotations)
        makeGTBox img scale ann =
            let (x0, y0, x1, y1) = cleanBBox img (ann ^. ann_bbox)
                classId = catTabl M.! (ann ^. ann_category_id)
            in
            if ann ^. ann_area > 0 && x1 > x0 && y1 > y0
              then Just $ fromListUnboxed (Z :. 5) [x0*scale, y0*scale, x1*scale, y1*scale, fromIntegral classId]
              else Nothing

    width <- view conf_width

    let imgFilePath = base </> datasplit </> img ^. img_file_name
    imgRGB <- raiseLeft (FileNotFound imgFilePath) <$> liftIO (HIP.readImage imgFilePath)

    let (imgH, imgW) = HIP.dims (imgRGB :: HIP.Image HIP.VS HIP.RGB Double)
        imgH_  = fromIntegral imgH
        imgW_  = fromIntegral imgW
        width_ = fromIntegral width
        (scale, imgW', imgH') = if imgW >= imgH
            then (width_ / imgW_, width, floor (imgH_ * width_ / imgW_))
            else (width_ / imgH_, floor (imgW_ * width_ / imgH_), width)
        imgInfo = fromListUnboxed (Z :. 3) [fromIntegral imgH', fromIntegral imgW', scale]

        imgResized = HIP.resize HIP.Bilinear HIP.Edge (imgH', imgW') imgRGB
        imgPadded  = HIP.canvasSize (HIP.Fill $ HIP.PixelRGB 0 0 0) (width, width) imgResized
        imgRepa    = Repa.fromUnboxed (Z:.width:.width:.3) $ SV.convert $ SV.unsafeCast $ HIP.toVector imgPadded
        gt_boxes   = get_gt_boxes scale img

    if V.null gt_boxes
        then return Nothing
        else do
            imgEval <- transform $ Repa.map double2Float imgRepa
            -- deepSeq the array so that the workload are well parallelized.
            return $!! Just (img ^. img_file_name, Repa.computeUnboxedS imgEval, imgInfo, gt_boxes)
  where
    cleanBBox img (x, y, w, h) =
        let width   = img ^. img_width
            height  = img ^. img_height
            x0 = max 0 x
            y0 = max 0 y
            x1 = min (fromIntegral width - 1)  (x0 + max 0 (w-1))
            y1 = min (fromIntegral height - 1) (y0 + max 0 (h-1))
        in (x0, y0, x1, y1)

