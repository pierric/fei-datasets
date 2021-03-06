{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures    #-}
{-# LANGUAGE OverloadedLabels  #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards   #-}
module Main where

import           Codec.Picture.Types
import           Control.Monad.Trans.Resource
import           Data.Attoparsec.Text         (char, decimal, endOfInput,
                                               parseOnly, rational, sepBy)
import           Data.Conduit
import qualified Data.Conduit.Combinators     as C (map, mapM, mapM_, take)
import qualified Data.Conduit.List            as C (catMaybes)
import           Data.Random.Source.StdGen    (mkStdGen)
import qualified Data.Vector.Storable         as SV (unsafeCast)
import           GHC.TypeLits                 (Symbol)
import qualified Graphics.Image               as HIP
import qualified Graphics.Image.Interface     as HIP
import           Graphics.Rasterific
import           Graphics.Rasterific.Texture  (uniformTexture)
import           Graphics.Text.TrueType
import           Options.Applicative          (Parser, auto, eitherReader,
                                               execParser, fullDesc, header,
                                               help, helper, info, long,
                                               metavar, option, showDefault,
                                               strOption, switch, value, (<**>))
import           RIO
import           RIO.FilePath
import qualified RIO.Text                     as T
import qualified RIO.Vector.Boxed             as V
import qualified RIO.Vector.Boxed.Partial     as V ((!))
import qualified RIO.Vector.Storable          as SV
import qualified RIO.Vector.Unboxed           as UV

import           MXNet.Base                   hiding (Symbol)
import           MXNet.Base.Tensor            (cast, mulScalar, sliceAxis)
import qualified MXNet.NN.DataIter.Coco       as DC
import           MXNet.NN.DataIter.Common
import qualified MXNet.NN.DataIter.PascalVOC  as DV

data Args = Args
    { arg_dataset   :: String
    , arg_base_dir  :: String
    , arg_datasplit :: String
    , arg_width     :: Int
    , arg_mean      :: (Float, Float, Float)
    , arg_stddev    :: (Float, Float, Float)
    , arg_num_imgs  :: Int
    , arg_shuffle   :: Maybe Int
    }

cmdArgParser = Args <$> strOption        (long "dataset" <> metavar "DATASET" <> help "dataset name")
                    <*> strOption        (long "base-dir" <> metavar "BASE" <> help "path to the root directory")
                    <*> strOption        (long "datasplit" <> metavar "SPLIT" <> help "datasplit")
                    <*> option auto      (long "img-size" <> metavar "SIZE" <> showDefault <> value 512 <> help "size of image")
                    <*> option floatList (long "img-pixel-means" <> metavar "RGB-MEAN" <> showDefault <> value (0.4850, 0.4580, 0.4076) <> help "RGB mean of images")
                    <*> option floatList (long "img-pixel-stddev" <> metavar "RGB-STD" <> showDefault <> value (1,1,1) <> help "RGB std-dev of images")
                    <*> option auto      (long "num-imgs" <> metavar "NUM-IMG" <> showDefault <> value 10 <> help "number of images")
                    <*> option auto      (long "shuffle" <> showDefault <> help "shuffle")
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

class HasWidth (a :: Symbol) where
    targetWidth :: Getting Int (Configuration a) Int

instance HasWidth "voc" where
    targetWidth = DV.conf_width

instance HasWidth "coco" where
    targetWidth = DC.conf_width

renderWithBBox :: ( HasCallStack
                  , HasWidth s
                  , HasDatasetConfig (Configuration s)
                  , ImageDataset (DatasetTag (Configuration s))
                  , MonadReader (Configuration s) m, MonadIO m)
                => Font
                -> (String, V.Vector String, ImageTensor, ImageInfo, GTBoxes)
                -> m (String, HIP.Image HIP.VS HIP.RGBA HIP.Word8)
renderWithBBox font (ident, cls, img, info, gt) = do
    width <- view targetWidth
    arr   <- transformInv img
    rawSV <- liftIO $ mulScalar 255 arr >>= cast #uint8 >>= toVector
    boxes <- liftIO $ V.unfoldr split <$> toVector gt
    let height = width
        img = promoteImage $ HIP.toJPImageRGB8 $ HIP.fromVector (height, width) $ SV.unsafeCast (rawSV :: SV.Vector Word8)
        res = renderDrawing width height (PixelRGBA8 0 0 0 0) $ do
                drawImage img 0 (V2 0 0)
                withTexture (uniformTexture $ PixelRGBA8 255 0 0 255) $ do
                    void $ forM (zip [0..] $ V.toList boxes) $ \(ind, gt) -> do
                        let [x0, y0, x1, y1, _] = SV.toList gt
                        stroke 1 JoinRound (CapRound, CapRound) $ rectangle (V2 x0 y0) (x1 - x0) (y1 - y0)
                        withTexture (uniformTexture $ PixelRGBA8 255 255 255 255) $ do
                            printTextAt font (PointSize 10) (V2 (x0+2) (y0+12)) (cls V.! ind)
    return $ (ident, HIP.fromJPImageRGBA8 res)

    where
        split vec | SV.null vec = Nothing
                  | otherwise   = Just $ SV.splitAt 5 vec

lookupClassName :: (HasCallStack, MonadIO m)
                => V.Vector String
                -> (String, ImageTensor, ImageInfo, GTBoxes)
                -> m (String, V.Vector String, ImageTensor, ImageInfo, GTBoxes)
lookupClassName table (imgname, tensor, info, gt) = liftIO $ do
    cls <- sliceAxis gt 1 4 (Just 5) >>= toVector
    let names = V.map (table V.!) $ SV.convert $ SV.map floor cls
    return (imgname, names, tensor, info, gt)

main :: IO ()
main = do
    fontCache <- buildCache
    let fontPath = findFontInCache fontCache (FontDescriptor "Hack" (FontStyle False False))
    fontPath <- case fontPath of
        Nothing -> error "font not found"
        Just a  -> return a
    font <- loadFontFile fontPath
    font <- case font of
        Left msg -> error msg
        Right a  -> return a
    Args{..} <- execParser $ info (cmdArgParser <**> helper) fullDesc
    let save (ident, img) = liftIO $ HIP.writeImageExact HIP.PNG [] (ident <.> "png") img
        dump :: (HasWidth s,  HasDatasetConfig (Configuration s), ImageDataset (DatasetTag (Configuration s)), MonadReader (Configuration s) m, MonadIO m) =>
                ConduitT (String, V.Vector String, ImageTensor, ImageInfo, GTBoxes) Void m ()
        dump = C.take arg_num_imgs .|
               C.mapM (renderWithBBox font) .|
               C.mapM_ save

    rand_gen <- newIORef $ mkStdGen $ fromMaybe 0 arg_shuffle

    case arg_dataset of
        "coco" -> do
            coco <- DC.coco arg_base_dir arg_datasplit
            let conf = DC.CocoConfig coco arg_width arg_mean arg_stddev
                iter = DC.cocoImagesBBoxes rand_gen .| C.mapM (DC.augmentWithBBoxes rand_gen)
            void $ runResourceT $ flip runReaderT conf $ runConduit $ iter .| C.mapM (lookupClassName DC.classes) .| dump
        "voc" -> do
            let conf = DV.VOCConfig arg_base_dir arg_width arg_mean arg_stddev
                iter = DV.vocMainImages arg_datasplit rand_gen .| C.mapM DV.loadImageAndBBoxes .| C.catMaybes
            void $ flip runReaderT conf $ runConduit $ iter .| C.mapM (lookupClassName $ V.map T.unpack DV.classes) .| dump

