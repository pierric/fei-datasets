{-# LANGUAGE DataKinds #-}
module MXNet.NN.DataIter.Common where

import RIO
import qualified RIO.Vector.Boxed as V
import GHC.TypeLits (Symbol)
import Data.Array.Repa (Array, DIM1, DIM3, D, U, (:.)(..), Z (..),
    fromListUnboxed, (-^), (+^), (*^), (/^))
import qualified Data.Array.Repa as Repa

type ImageTensor = Array U DIM3 Float
type ImageInfo = Array U DIM1 Float
type GTBoxes = V.Vector (Array U DIM1 Float)

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
              Repa.Source r Float)
    => Array r DIM3 Float -> m (Array D DIM3 Float)
transform img = do
    mean <- view (datasetConfig . imagesMean)
    std  <- view (datasetConfig . imagesStdDev)
    let broadcast = Repa.extend (Repa.Any :. height :. width)
        mean' = broadcast $ fromTuple mean
        std'  = broadcast $ fromTuple std
        chnFirst = Repa.backpermute newShape (\ (Z :. c :. h :. w) -> Z :. h :. w :. c) img
    return $ (chnFirst -^ mean') /^ std'
  where
    Z :. height :. width :. chn = Repa.extent img
    newShape = Z:. chn :. height :. width

-- transform CHW -> HWC
transformInv :: (ImageDataset s, MonadReader (Configuration s) m, Repa.Source r Float) =>
    Array r DIM3 Float -> m (Array D DIM3 Float)
transformInv img = do
    mean <- view imagesMean
    std <- view imagesStdDev
    let broadcast = Repa.extend (Repa.Any :. height :. width)
        mean' = broadcast $ fromTuple mean
        std'  = broadcast $ fromTuple std
        addMean = img *^ std' +^ mean'
    return $ Repa.backpermute newShape (\ (Z :. h :. w :. c) -> Z :. c :. h :. w) addMean
  where
    (Z :. chn :. height :. width) = Repa.extent img
    newShape = Z :. height :. width :. chn

fromTuple (a, b, c) = fromListUnboxed (Z :. (3 :: Int)) [a,b,c]

raiseLeft :: (MonadThrow m, Exception e) => (a -> e) -> m (Either a b) -> m b
raiseLeft exc act = act >>= either (throwM . exc) return

instance (Repa.Shape sh, Unbox e) => NFData (Array U sh e) where
    rnf arr = Repa.deepSeqArray arr ()

