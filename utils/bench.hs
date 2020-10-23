import qualified Codec.Picture                as JUC
import qualified Codec.Picture.Extra          as JUC
import qualified Codec.Picture.Repa           as RPJ
import           Control.Monad.Trans.Resource
import           Criterion.Main
import qualified Data.Array.Repa              as Repa
import           Data.Conduit
import           Data.Conduit.Async
import qualified Data.Conduit.List            as C
import           Data.Random.Source.StdGen    (mkStdGen)
import qualified Graphics.Image               as HIP
import           RIO
import           RIO.Directory
import           RIO.FilePath

import qualified MXNet.NN.DataIter.Coco       as Coco

main = do
    home <- getHomeDirectory
    let imgFilePath = home </> ".mxnet/datasets/coco/val2017/000000121242.jpg"
    Right imgjuc <- liftIO (JUC.readImage imgFilePath)
    Right imghip <- liftIO (HIP.readImage imgFilePath :: IO (Either String (HIP.Image HIP.VS HIP.RGB Double)))

    rand_gen <- newIORef $ mkStdGen 99
    cocoInst <- Coco.coco (home </> ".mxnet/datasets/coco") "val2017"
    let cocoConf = Coco.CocoConfig cocoInst 1024 (0.5, 0.5, 0.5) (1, 1, 1)
        iter1 = Coco.cocoImages rand_gen
        iter2 = Coco.cocoImagesBBoxes rand_gen

    defaultMain
        [
          bench "scale-img-juicy" $ nfIO $
            let img1 = JUC.convertRGB8 imgjuc
                img2 = JUC.scaleBilinear 1024 1024 img1
                img3 = RPJ.imgData (RPJ.convertImage img2 :: RPJ.Img RPJ.RGB)
            in Repa.computeUnboxedP img3

        , bench "scale-img-hip" $ nfIO $
            let img2 = HIP.resize HIP.Bilinear HIP.Edge (1024, 1024) imghip
            in return img2

        , bench "img-iter" $ nfIO $
            runResourceT $
                flip runReaderT cocoConf $
                    runConduit $ iter1 .| C.take 10

        , bench "img-iter-async" $ nfIO $
            runResourceT $
                flip runReaderT cocoConf $
                    runCConduit $ iter1 =$=& C.take 10

        , bench "img-iter-and-load-gt" $ nfIO $
            runResourceT $
                flip runReaderT cocoConf $
                    runConduit $ iter2 .| C.take 10
        ]
