{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
module Main where

import GHC.TypeLits (Symbol)
import Options.Applicative (
    Parser, execParser,
    long, value, option, auto, strOption, switch, metavar, showDefault, eitherReader, help,
    info, helper, fullDesc, header, (<**>))
import Data.Attoparsec.Text (sepBy, char, rational, decimal, endOfInput, parseOnly)
import Data.Conduit
import qualified Data.Conduit.Combinators as C (mapM, mapM_, take, map)
import qualified Data.Conduit.List as C (catMaybes)
import qualified Data.Text as T
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Unboxed as UV
import qualified Data.Array.Repa as Repa
import Data.Array.Repa ((:.)(..), Z (..))
import Control.Applicative
import Control.Monad.Reader
import Control.Monad.Trans.Resource
import Control.Lens (view, Getting)
import qualified Graphics.Image as HIP
import qualified Graphics.Image.Interface as HIP
import Graphics.Rasterific
import Graphics.Rasterific.Texture (uniformTexture)
import Graphics.Text.TrueType
import Codec.Picture.Types
import System.FilePath
import MXNet.NN.DataIter.Common
import qualified MXNet.NN.DataIter.PascalVOC as DV
import qualified MXNet.NN.DataIter.Coco as DC

import Debug.Trace

data Args = Args {
    arg_dataset :: String,
    arg_base_dir :: String,
    arg_datasplit :: String,
    arg_width :: Int,
    arg_mean :: (Float, Float, Float),
    arg_stddev :: (Float, Float, Float),
    arg_num_imgs :: Int,
    arg_shuffle :: Bool
}

cmdArgParser = Args <$> strOption        (long "dataset" <> metavar "DATASET" <> help "dataset name")
                    <*> strOption        (long "base-dir" <> metavar "BASE" <> help "path to the root directory")
                    <*> strOption        (long "datasplit" <> metavar "SPLIT" <> help "datasplit")
                    <*> option auto      (long "img-size" <> metavar "SIZE" <> showDefault <> value 512 <> help "size of image")
                    <*> option floatList (long "img-pixel-means" <> metavar "RGB-MEAN" <> showDefault <> value (0.4850, 0.4580, 0.4076) <> help "RGB mean of images")
                    <*> option floatList (long "img-pixel-stddev" <> metavar "RGB-STD" <> showDefault <> value (1,1,1) <> help "RGB std-dev of images")
                    <*> option auto      (long "num-imgs" <> metavar "NUM-IMG" <> showDefault <> value 10 <> help "number of images")
                    <*> switch           (long "shuffle" <> showDefault <> help "shuffle")
  where
    triple = do
        a <- rational
        char ','
        b <- rational
        char ','
        c <- rational
        endOfInput
        return (a, b, c)
    floatList = eitherReader (parseOnly triple . T.pack)

class HasImageWidth (a :: Symbol) where
    targetWidth :: Getting Int (Configuration a) Int

instance HasImageWidth "voc" where
    targetWidth = DV.conf_width

instance HasImageWidth "coco" where
    targetWidth = DC.conf_width

renderWithBBox :: (HasImageWidth s, ImageDataset s, MonadReader (Configuration s) m, MonadIO m) =>
    Font -> (String, V.Vector String, ImageTensor, ImageInfo, GTBoxes) -> m (String, HIP.Image HIP.VS HIP.RGBA HIP.Word8)
renderWithBBox font (ident, cls, img, info, gt) = do
    width <- view targetWidth
    let height = width
    arr <- transformInv img
    let rawUV = Repa.toUnboxed $ Repa.computeUnboxedS $ Repa.map (floor . (* 255.0)) arr :: UV.Vector HIP.Word8
        rawSV = SV.unsafeCast $ UV.convert rawUV  :: HIP.Vector HIP.VS (HIP.Pixel HIP.RGB HIP.Word8)
        img = promoteImage $ HIP.toJPImageRGB8 $ HIP.fromVector (height, width) rawSV
        res = renderDrawing width height (PixelRGBA8 0 0 0 0) $ do
                drawImage img 0 (V2 0 0)
                withTexture (uniformTexture $ PixelRGBA8 255 0 0 255) $ do
                    void $ forM (zip [0..] $ V.toList boxes) $ \(ind, [x0, y0, x1, y1, _]) -> do
                        stroke 1 JoinRound (CapRound, CapRound) $ rectangle (V2 x0 y0) (x1 - x0) (y1 - y0)
                        withTexture (uniformTexture $ PixelRGBA8 255 255 255 255) $ do
                            printTextAt font (PointSize 10) (V2 (x0+2) (y0+12)) (cls V.! ind)
    return $ (ident, HIP.fromJPImageRGBA8 res)
  where
    boxes = V.map (UV.toList . Repa.toUnboxed) gt

lookupClassName :: V.Vector String -> (String, ImageTensor, ImageInfo, GTBoxes) -> (String, V.Vector String, ImageTensor, ImageInfo, GTBoxes)
lookupClassName table (imgname, tensor, info, gt) = (imgname, gtNames, tensor, info, gt)
  where
    gtNames = V.map ((table V.!) . floor . (`Repa.index` (Z:.4))) gt

main :: IO ()
main = do
    fontCache <- buildCache
    let fontPath = findFontInCache fontCache (FontDescriptor "Hack" (FontStyle False False))
    fontPath <- case fontPath of
        Nothing -> error "font not found"
        Just a -> return a
    font <- loadFontFile fontPath
    font <- case font of
        Left msg -> error msg
        Right a -> return a
    Args{..} <- execParser $ info (cmdArgParser <**> helper) fullDesc
    let save (ident, img) = liftIO $ HIP.writeImageExact HIP.PNG [] (ident <.> "png") img
        dump :: (HasImageWidth s, ImageDataset s, MonadReader (Configuration s) m, MonadIO m) =>
                ConduitT (String, V.Vector String, ImageTensor, ImageInfo, GTBoxes) Void m ()
        dump = C.take arg_num_imgs .|
               C.mapM (renderWithBBox font) .|
               C.mapM_ save
    case arg_dataset of
        "coco" -> do
            coco <- DC.coco arg_base_dir arg_datasplit
            let conf = DC.CocoConfig coco arg_width arg_mean arg_stddev
                iter = DC.cocoImagesAndBBoxes arg_shuffle
            void $ runResourceT $ flip runReaderT conf $ runConduit $ iter .| C.map (lookupClassName DC.classes) .| dump
        "voc" -> do
            let conf = DV.VOCConfig arg_base_dir arg_width arg_mean arg_stddev
                iter = DV.vocMainImages arg_datasplit arg_shuffle .| C.mapM DV.loadImageAndBBoxes .| C.catMaybes
            void $ flip runReaderT conf $ runConduit $ iter .| C.map (lookupClassName DV.classes) .| dump

