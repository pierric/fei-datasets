{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds #-}
module MXNet.NN.DataIter.PascalVOC (
    module MXNet.NN.DataIter.Common,
    Configuration(..), VOCConfig, conf_width,
    classes, vocMainImages, loadImageAndBBoxes
) where
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.ByteString as B
import Data.Maybe (catMaybes)
import Text.XML.Expat.Proc
import Text.XML.Expat.Tree
import Data.Conduit
import qualified Data.Conduit.List as C
import Data.Array.Repa (Array, DIM1, DIM3, U, D, (:.)(..), Z (..),
    fromListUnboxed, (-^), (+^), (*^), (/^))
import qualified Data.Array.Repa as Repa
import Data.Array.Repa.Repr.Unboxed (Unbox)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Data.Conduit.Combinators as C (yieldMany)
import qualified Data.Random as RND (shuffleN, runRVar, StdRandom(..))
import qualified Graphics.Image as HIP
import qualified Graphics.Image.Interface as HIP
import Control.Lens ((^.), view, makeLenses)
import Control.Monad.Reader (MonadReader)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.DeepSeq (($!!), NFData(..))
import Control.Exception (Exception, throw)
import System.FilePath
import GHC.Float (double2Float)

import MXNet.NN.DataIter.Common

data Exc = FileNotFound String String | CannotParseAnnotation String
  deriving Show
instance Exception Exc

classes :: V.Vector String
classes = V.fromList [
    "__background__",  -- always index 0
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"]

data instance Configuration "voc" = VOCConfig {
    _conf_base_dir :: FilePath,
    _conf_width :: Int,
    _conf_mean :: (Float, Float, Float),
    _conf_std :: (Float, Float, Float)
}
makeLenses 'VOCConfig

type VOCConfig = Configuration "voc"

instance ImageDataset "voc" where
    imagesMean = conf_mean
    imagesStdDev = conf_std

vocMainImages :: (MonadReader VOCConfig m, MonadIO m) =>
    String -> Bool -> ConduitT () String m ()
vocMainImages datasplit shuffle = do
    base <- view conf_base_dir
    let imageset = base </> "ImageSets" </> "Main" </> datasplit <.> "txt"
    content <- liftIO $ T.readFile imageset
    let image_list = T.lines content
    all_images <- if shuffle then
                    liftIO $ RND.runRVar (RND.shuffleN (length image_list) image_list) RND.StdRandom
                  else
                    return $ image_list
    C.yieldMany all_images .| C.map T.unpack

loadImageAndBBoxes :: (MonadReader VOCConfig m, MonadIO m) =>
    String -> m (Maybe (String, ImageTensor, ImageInfo, GTBoxes))
loadImageAndBBoxes ident = do
    width <- view conf_width
    base <- view conf_base_dir

    let imgFilePath = base </> "JPEGImages" </> ident <.> "jpg"
    imgRGB <- raiseLeft (FileNotFound imgFilePath) <$> liftIO
        (HIP.readImageExact HIP.JPG imgFilePath)

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

    let annoFilePath = base </> "Annotations" </> ident <.> "xml"
    xml <- liftIO $ B.readFile annoFilePath
    gtBoxes <- case parse' defaultParseOptions xml of
        Left err -> throw (CannotParseAnnotation annoFilePath) err
        Right root -> do
            let objs = findElements "object" root
            return $ V.fromList $ catMaybes $ map (makeGTBox scale) objs

    if V.null gtBoxes
        then return Nothing
        else do
            imgEval <- transform $ Repa.map double2Float imgRepa
            -- deepSeq the array so that the workload are well parallelized.
            return $!! Just (ident, Repa.computeUnboxedS imgEval, imgInfo, gtBoxes)
  where
    makeGTBox scale node = do
        className <- textContent <$> findElement "name" node
        bndbox <- findElement "bndbox" node
        xmin <- textContent <$> findElement "xmin" bndbox
        xmax <- textContent <$> findElement "xmax" bndbox
        ymin <- textContent <$> findElement "ymin" bndbox
        ymax <- textContent <$> findElement "ymax" bndbox
        classId <- V.elemIndex className classes
        let x0 = read xmin
            x1 = read xmax
            y0 = read ymin
            y1 = read ymax
        return $ fromListUnboxed (Z :. 5) [x0*scale, y0*scale, x1*scale, y1*scale, fromIntegral classId]
