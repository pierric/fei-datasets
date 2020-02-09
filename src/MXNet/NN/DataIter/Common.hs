{-# LANGUAGE DataKinds #-}
module MXNet.NN.DataIter.Common where

import GHC.TypeLits (Symbol)
import Data.Array.Repa (Array, DIM1, DIM3, D, U, (:.)(..), Z (..), Any(..),
    fromListUnboxed, (-^), (+^), (*^), (/^))
import qualified Data.Array.Repa as Repa
import Data.Array.Repa.Repr.Unboxed (Unbox)
import qualified Data.Vector as V
import Control.Lens ((^.), view, makeLenses, Getting)
import Control.DeepSeq
import Control.Exception
import Control.Monad.Reader

type ImageTensor = Array U DIM3 Float
type ImageInfo = Array U DIM1 Float
type GTBoxes = V.Vector (Array U DIM1 Float)

data family Configuration (dataset :: Symbol)

class ImageDataset (a :: Symbol) where
    imagesMean :: Getting (Float, Float, Float) (Configuration a) (Float, Float, Float)
    imagesStdDev :: Getting (Float, Float, Float) (Configuration a) (Float, Float, Float)

-- transform HWC -> CHW
transform :: (ImageDataset s, MonadReader (Configuration s) m, Repa.Source r Float) =>
    Array r DIM3 Float -> m (Array D DIM3 Float)
transform img = do
    mean <- view imagesMean
    std <- view imagesStdDev
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

raiseLeft :: Exception e => (a -> e) -> Either a b -> b
raiseLeft exc = either (throw . exc) id

instance (Repa.Shape sh, Unbox e) => NFData (Array U sh e) where
    rnf arr = Repa.deepSeqArray arr ()

