{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE UndecidableInstances #-}
module MXNet.NN.DataIter.PascalVOC (
    module MXNet.NN.DataIter.Common,
    Configuration(..), VOCConfig, conf_width,
    classes, vocMainImages, loadImageAndBBoxes
) where

import           Control.Exception         (throw)
import           Control.Lens              (makeLenses)
import           Data.Conduit
import qualified Data.Conduit.Combinators  as C (yieldMany)
import qualified Data.Conduit.List         as C
import qualified Data.Random               as RND (runRVar, shuffleN,
                                                   stdUniform)
import           Data.Random.Source.StdGen (StdGen)
import qualified Data.Vector.Storable      as SV (unsafeCast)
import           GHC.Float                 (double2Float)
import qualified Graphics.Image            as HIP
import qualified Graphics.Image.Interface  as HIP
import           RIO
import qualified RIO.ByteString            as B
import           RIO.FilePath
import qualified RIO.Text                  as T
import qualified RIO.Vector.Boxed          as V
import qualified RIO.Vector.Storable       as SV
import           Text.XML.Expat.Proc
import           Text.XML.Expat.Tree

import           MXNet.Base
import           MXNet.Base.ParserUtils    (parseR, rational)
import           MXNet.Base.Tensor         (cast, stack)
import           MXNet.NN.DataIter.Common

data Exc = FileNotFound String String
    | CannotParseAnnotation String
    deriving Show
instance Exception Exc

classes :: V.Vector Text
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

instance HasDatasetConfig VOCConfig where
    type DatasetTag VOCConfig = "voc"
    datasetConfig = id

instance ImageDataset "voc" where
    imagesMean = conf_mean
    imagesStdDev = conf_std

vocMainImages :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "voc", MonadIO m) =>
    String -> IORef StdGen -> ConduitT () String m ()
vocMainImages datasplit rand_gen = do
    base <- view (datasetConfig . conf_base_dir)
    let imageset = base </> "ImageSets" </> "Main" </> datasplit <.> "txt"
    content <- readFileUtf8 imageset
    let image_list = T.lines content
    all_images <- liftIO $ RND.runRVar (RND.shuffleN (length image_list) image_list) rand_gen
    C.yieldMany all_images .| C.map T.unpack

loadImageAndBBoxes :: (MonadReader env m, HasDatasetConfig env, DatasetTag env ~ "voc", MonadIO m)
    => String -> m (Maybe (String, ImageTensor, ImageInfo, GTBoxes))
loadImageAndBBoxes ident = do
    width <- view (datasetConfig . conf_width)
    base <- view (datasetConfig . conf_base_dir)

    let imgFilePath = base </> "JPEGImages" </> ident <.> "jpg"
    imgRGB <- liftIO $ raiseLeft (FileNotFound imgFilePath) $
        (HIP.readImageExact HIP.JPG imgFilePath)

    let (oriH, oriW) = HIP.dims (imgRGB :: HIP.Image HIP.VS HIP.RGB Double)
        (scale, imgH, imgW) = getImageScale oriH oriW width
        imgResized = HIP.resize HIP.Bilinear HIP.Edge (imgH, imgW) (imgRGB :: HIP.Image HIP.VS HIP.RGB Double)
        imgPadded  = HIP.canvasSize (HIP.Fill $ HIP.PixelRGB 0.5 0.5 0.5) (width, width) imgResized

    info <- liftIO $ fromVector [3] [fromIntegral imgH, fromIntegral imgW, scale]
    let img_padded_vec = SV.unsafeCast $ HIP.toVector imgPadded :: SV.Vector Double
    img_f <- liftIO $ fromVector [width, width, 3] img_padded_vec >>= cast #float32
    img_f <- transform img_f

    let annoFilePath = base </> "Annotations" </> ident <.> "xml"
    xml <- liftIO $ B.readFile annoFilePath
    gtBoxes <- case parse' defaultParseOptions xml of
        Left err -> throw (CannotParseAnnotation annoFilePath) err
        Right root -> do
            let objs = findElements "object" (root :: Node Text Text)
            return $ catMaybes $ map (makeGTBox scale) objs

    if null gtBoxes
    then return Nothing
    else liftIO $ do
        gts <- mapM (fromVector [5]) gtBoxes
        gts <- stack 0 gts
        return $!! Just (ident, img_f, info, gts)

  where
    makeGTBox scale node = do
        className <- textContent <$> findElement "name" node
        bndbox <- findElement "bndbox" node
        xmin <- textContent <$> findElement "xmin" bndbox
        xmax <- textContent <$> findElement "xmax" bndbox
        ymin <- textContent <$> findElement "ymin" bndbox
        ymax <- textContent <$> findElement "ymax" bndbox
        classId <- V.elemIndex className classes
        let x0 = parseR rational xmin
            x1 = parseR rational xmax
            y0 = parseR rational ymin
            y1 = parseR rational ymax
        return $ [x0*scale, y0*scale, x1*scale, y1*scale, fromIntegral classId]

