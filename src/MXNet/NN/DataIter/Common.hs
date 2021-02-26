{-# LANGUAGE DataKinds #-}
module MXNet.NN.DataIter.Common where

import           Control.Lens        (each, (^..))
import           GHC.TypeLits        (Symbol)
import           RIO
import qualified RIO.Vector.Storable as SV

import           Fei.Einops
import           MXNet.Base          hiding (Symbol)
import           MXNet.Base.Tensor   (addBroadcast, divBroadcast, mulBroadcast,
                                      subBroadcast)

type ImageTensor = NDArray Float -- (H, W, 3)
type ImageInfo   = NDArray Float -- (3,)
type Anchors     = NDArray Float -- (M, 4)
type GTBoxes     = NDArray Float -- (N, 5)
type Masks       = NDArray Float -- (N, H, W)

data family Configuration (dataset :: Symbol)

class ImageDataset (a :: Symbol) where
    imagesMean :: Getting (Float, Float, Float) (Configuration a) (Float, Float, Float)
    imagesStdDev :: Getting (Float, Float, Float) (Configuration a) (Float, Float, Float)

class HasDatasetConfig env where
    type DatasetTag env :: Symbol
    datasetConfig :: Lens' env (Configuration (DatasetTag env))

-- transform HWC -> CHW
transform :: (HasDatasetConfig env,
              ImageDataset (DatasetTag env),
              MonadReader env m,
              MonadIO m)
          => NDArray Float -> m (NDArray Float)
transform img = do
    mean <- view (datasetConfig . imagesMean)
    std  <- view (datasetConfig . imagesStdDev)
    liftIO $ do
        std  <- fromVector [3, 1, 1] $ SV.fromList $ std  ^.. each
        mean <- fromVector [3, 1, 1] $ SV.fromList $ mean ^.. each
        imgCHW <- rearrange img "h w c -> c h w" []
        imgRet <- subBroadcast imgCHW mean
        imgRet <- divBroadcast imgRet std
        return imgRet

-- transform CHW -> HWC
transformInv :: (HasDatasetConfig env,
                 ImageDataset (DatasetTag env),
                 MonadReader env m,
                 MonadIO m)
             => NDArray Float -> m (NDArray Float)
transformInv img = do
    mean <- view (datasetConfig . imagesMean)
    std  <- view (datasetConfig . imagesStdDev)
    liftIO $ do
        mean <- fromVector [3, 1, 1] $ SV.fromList $ mean ^.. each
        std  <- fromVector [3, 1, 1] $ SV.fromList $ std  ^.. each
        img  <- mulBroadcast img std
        img  <- addBroadcast img mean
        img  <- rearrange img "c h w -> h w c" []
        return img

raiseLeft :: (MonadThrow m, Exception e) => (a -> e) -> m (Either a b) -> m b
raiseLeft exc act = act >>= either (throwM . exc) return

getImageScale :: Int -> Int -> Int -> (Float, Int, Int)
getImageScale height width size
  | width >= height = (sizeF / oriW, floor (oriH* sizeF / oriW), size)
  | otherwise       = (sizeF / oriH, size, floor (oriW* sizeF / oriH))
    where
        oriW  = fromIntegral width
        oriH  = fromIntegral height
        sizeF = fromIntegral size
