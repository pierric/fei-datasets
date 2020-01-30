module MXNet.NN.DataIter.VOC where

import qualified Data.Text as T
import Data.Conduit
import qualified Data.Conduit.Combinators as C (yieldMany)
import qualified Data.Conduit.List as C


type ImageTensor = Array U DIM3 Float
type ImageInfo = Array U DIM1 Float
type GTBoxes = V.Vector (Array U DIM1 Float)

vocMainImages :: FilePath -> String -> ConduitT () T.Text m ()
vocMainImages base datasplit = do
    content <- T.readFile imageset
    C.yieldMany $ T.unlines content
  where
    imageset = base </> "ImageSets" </> "Main" </> ^. "txt"


loadImageAndGT :: (MonadReader Configuration m, MonadIO m) => FilePath -> String -> m (Maybe (ImageTensor, ImageInfo, GTBoxes))
loadImageAndGT base ident = do
    width <- view conf_width

    let imgFilePath = base </> "JPEGImages" </> ident ^. "jpg"
    imgRGB <- raiseLeft (FileNotFound imgFilePath) <$> liftIO 
        (HIP.readImageExact HIP.JPG imgFilePath)

    let annoFilePath = base </> "Annotations" </> ident ^. "xml"

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
            return $!! Just (Repa.computeUnboxedS imgEval, imgInfo, gt_boxes)
  where
    -- map each category from id to its index in the cocoClassNames.
    catTabl = M.fromList $ V.toList $ V.map (\cat -> (cat ^. odc_id, fromJust $ V.elemIndex (cat ^. odc_name) cocoClassNames)) (inst ^. categories)

    -- get all the bbox and gt for the image
    get_gt_boxes scale img = V.fromList $ catMaybes $ map makeGTBox $ V.toList imgAnns
      where
        imageId = img ^. img_id
        width   = img ^. img_width
        height  = img ^. img_height
        imgAnns = V.filter (\ann -> ann ^. ann_image_id == imageId) (inst ^. annotations)

        cleanBBox (x, y, w, h) =
          let x0 = max 0 x
              y0 = max 0 y
              x1 = min (fromIntegral width - 1)  (x0 + max 0 (w-1))
              y1 = min (fromIntegral height - 1) (y0 + max 0 (h-1))
          in (x0, y0, x1, y1)

        makeGTBox ann =
          let (x0, y0, x1, y1) = cleanBBox (ann ^. ann_bbox)
              classId = catTabl M.! (ann ^. ann_category_id)

          in
          if ann ^. ann_area > 0 && x1 > x0 && y1 > y0
            then Just $ fromListUnboxed (Z :. 5) [x0*scale, y0*scale, x1*scale, y1*scale, fromIntegral classId]
            else Nothing
