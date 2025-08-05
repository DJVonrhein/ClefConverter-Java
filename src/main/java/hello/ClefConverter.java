package ClefConverter;


import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;  
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
// import org.joda.time.LocalTime;

public class ClefConverter {
    static {
        nu.pattern.OpenCV.loadLocally();
        // nu.pattern.OpenCV.loadLocally(); // Use in case loadShared() doesn't work
    }
    public static Mat getTemplateMatchesMask(Mat img, Mat template, double threshold ){
        // img: matrix to search 
        // template: image of interest
        // threshold: a passing score

        Mat matches = new Mat();        //doubles version
        Imgproc.matchTemplate(img, template, matches, Imgproc.TM_CCOEFF_NORMED) ;

        Mat matchesMask = new Mat();    //binary version
        Imgproc.threshold(matches, matchesMask, threshold, 255, Imgproc.THRESH_BINARY);
        matchesMask.convertTo(matchesMask, CvType.CV_8UC1);
        return matchesMask;
    }
    public static Mat coverTemplateMatchWithStructuringElement(Mat templateMatchBinary, Mat structuringElement){
        // templateMatchBinary:  a mask showing where a template was matched
        // structuringElement :  a cover to place on each template match
        // output: mask after covering every white pixel to structuringElement, then upsizing it to its original image 

        
        Imgcodecs.imwrite("./img/before.png", templateMatchBinary);
        Core.bitwise_not(templateMatchBinary, templateMatchBinary);
        
        Mat paddedMatches = new Mat();
        Imgproc.erode(templateMatchBinary, templateMatchBinary, structuringElement);
        // Core.copyMakeBorder​(templateMatchBinary, paddedMatches, structuringElement.rows() - 1, 0 , structuringElement.cols() - 1, 0,  Core.BORDER_CONSTANT, new Scalar(255));
        int top = structuringElement.rows() / 2;
        int bottom = top;
        int left = structuringElement.cols() / 2;
        int right = left;
        if(structuringElement.rows()% 2 == 0) {
            top--;
        }
        if(structuringElement.cols() % 2 == 0){
            left--;
        }
        Core.copyMakeBorder​(templateMatchBinary, paddedMatches, top, bottom , left, right,  Core.BORDER_CONSTANT, new Scalar(255));

        Imgproc.erode(templateMatchBinary, templateMatchBinary, structuringElement);
        // Imgproc.erode(templateMatchBinary, templateMatchBinary, structuringElement, new Point(x, y), 1,  Core.BORDER_CONSTANT, new Scalar(255));
        
        Core.bitwise_not(paddedMatches, paddedMatches);
        Imgcodecs.imwrite("./img/after.png", paddedMatches);
        return paddedMatches;
    }
    public static Mat eraseShortRests(Mat img, int noteHeight, double threshold){
        Mat imgBefore = img.clone();

        Mat quarterRestTemplate = Imgcodecs.imread("./templates/rests/quarterrest.png", Imgcodecs.IMREAD_GRAYSCALE) ;
        // System.out.println("quarterRestTemplate shape:" + quarterRestTemplate.rows() + "   " + quarterRestTemplate.cols());
        System.out.println(CvType.typeToString(quarterRestTemplate.type()));
        int upHeight = (int)(noteHeight * 2.8);
        int upWidth = upHeight * quarterRestTemplate.cols() /quarterRestTemplate.rows() ;
        // System.out.println("new shape:" + upHeight + "    " + upWidth);
        Imgproc.resize(quarterRestTemplate, quarterRestTemplate, new Size(upWidth, upHeight), Imgproc.INTER_LINEAR);
        Mat quarterRestStructuringElement = new Mat(new Size(upWidth ,upHeight),  CvType.CV_8UC1, new Scalar(255)); 
        //TODO: ERASE FORTES PRIOR, diagonal dilation to recover 8th and 16th rests

        // Perform match operations - quarter rests need looser threshold
        Mat quarterRestMatchesMask = getTemplateMatchesMask(img, quarterRestTemplate, threshold + 0.06);
      
        //convert white pixels to treble sized rectangles
        quarterRestMatchesMask = coverTemplateMatchWithStructuringElement(quarterRestMatchesMask, quarterRestStructuringElement);

        //replace the bounding box with the treble inside
        Mat quarterRestMask = new Mat();
        Core.min(quarterRestMatchesMask, img, quarterRestMask);
        Core.bitwise_xor(quarterRestMask, quarterRestMatchesMask, quarterRestMask);


        Mat eighthRestTemplate = Imgcodecs.imread("./templates/rests/eighthrest.png", Imgcodecs.IMREAD_GRAYSCALE) ;
        upHeight = (int)(noteHeight * 2.15);
        upWidth = upHeight * eighthRestTemplate.cols()/eighthRestTemplate.rows() ;
        Imgproc.resize(eighthRestTemplate, eighthRestTemplate, new Size(upWidth, upHeight), Imgproc.INTER_LINEAR);
        Mat eighthRestStructuringElement = new Mat(new Size(upWidth ,upHeight),  CvType.CV_8UC1, new Scalar(255)); 

        // Perform match operations 
        Mat eighthRestMatchesMask = getTemplateMatchesMask(img, eighthRestTemplate, threshold);
      
        //convert white pixels to treble sized rectangles
        eighthRestMatchesMask = coverTemplateMatchWithStructuringElement(eighthRestMatchesMask, eighthRestStructuringElement);

        //replace the bounding box with the treble inside
        Mat eighthRestMask = new Mat();
        Core.min(eighthRestMatchesMask, img, eighthRestMask);
        Core.bitwise_xor(eighthRestMask, eighthRestMatchesMask, eighthRestMask);

        Mat sixteenthRestTemplate = Imgcodecs.imread("./templates/rests/sixteenthrest.png", Imgcodecs.IMREAD_GRAYSCALE) ;
        upHeight = (int)(noteHeight * 2.15);
        upWidth = upHeight * sixteenthRestTemplate.cols() /sixteenthRestTemplate.rows() ;
        Imgproc.resize(sixteenthRestTemplate, sixteenthRestTemplate, new Size(upWidth, upHeight), Imgproc.INTER_LINEAR);
        Mat sixteenthRestStructuringElement = new Mat(new Size(upWidth ,upHeight),  CvType.CV_8UC1, new Scalar(255)); 
        //TODO: fix "cutoff" appearance for 16th rests
        
        // Perform match operations 
        Mat sixteenthRestMatchesMask = getTemplateMatchesMask(img, sixteenthRestTemplate, threshold);
      
        //convert white pixels to treble sized rectangles
        sixteenthRestMatchesMask = coverTemplateMatchWithStructuringElement(sixteenthRestMatchesMask, sixteenthRestStructuringElement);

        //replace the bounding box with the treble inside
        Mat sixteenthRestMask = new Mat();
        Core.min(sixteenthRestMatchesMask, img, sixteenthRestMask);
        Core.bitwise_xor(sixteenthRestMask, sixteenthRestMatchesMask, sixteenthRestMask);

        Mat shortRestsMask = new Mat();
        quarterRestMask.copyTo(shortRestsMask);
        
        Core.max(shortRestsMask, eighthRestMask, shortRestsMask);
        Core.max(shortRestsMask, sixteenthRestMask, shortRestsMask);

        //TODO: base it off of a real variable (staff upper threshold)
        // Imgproc.threshold(imgBefore, imgBefore, 181, 255, Imgproc.THRESH_BINARY_INV);
        
        return shortRestsMask;
    }
    // expands clef areas vertically.  Only accounts for treble and bass (green and blue, respectively). This function is somewhat of a bottleneck with its python for loops.
    public static Mat addClefAreas(List<Integer> lineSeparatorsArray, Mat musicLines, Mat treblesMask, Mat bassesMask){
        long startTime  = System.currentTimeMillis();
        
        //temporarily downsize for efficiency
        Mat musicLinesShrunk = new Mat();
        Imgproc.resize(musicLines, musicLinesShrunk, new Size(musicLines.cols()/10, musicLines.rows()/10), Imgproc.INTER_AREA);
        int rows = musicLinesShrunk.rows();
        int cols = musicLinesShrunk.cols();
        int channels = musicLinesShrunk.channels();

        //eliminate new gray areas present after aveeraging in resize()
        //TODO: can resize() not do this?
        // Imgproc.threshold(musicLinesShrunk, musicLinesShrunk, 0, 255, Imgproc.THRESH_BINARY);
        // music_lines_shrunk[music_lines_shrunk > 0] = 255

        Mat treblesMaskShrunk = new Mat();
        Imgproc.resize(treblesMask, treblesMaskShrunk, new Size(cols, rows), Imgproc.INTER_AREA);
        // trebles_mask_shrunk = cv.resize(trebles_mask, (trebles_mask.shape[1]//10, trebles_mask.shape[0]//10))
        Mat bassesMaskShrunk = new Mat();
        Imgproc.resize(bassesMask, bassesMaskShrunk, new Size(cols, rows), Imgproc.INTER_AREA);
        // basses_mask_shrunk = cv.resize(basses_mask, (basses_mask.shape[1]//10, basses_mask.shape[0]//10))
        
        //eliminate new gray areas present after aveeraging in resize()
        //TODO: can resize() not do this?
        Imgproc.threshold(treblesMaskShrunk, treblesMaskShrunk, 0, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(bassesMaskShrunk, bassesMaskShrunk, 0, 255, Imgproc.THRESH_BINARY);

        // System.out.println("musicLines type: " + CvType.typeToString(musicLines.type()));

        // Imgcodecs.imwrite("./img/treblesMaskShrunk.png", treblesMaskShrunk);
        // Imgcodecs.imwrite("./img/bassesMaskShrunk.png", bassesMaskShrunk);

        
        Mat musicLinesColored = new Mat(musicLinesShrunk.size(), CvType.CV_8UC3, new Scalar(0));
        // System.out.println("channels: " + musicLinesColored.channels());
        int prevRow = lineSeparatorsArray.get(1) / 10;
        int row;

        for (int i = 2; i < lineSeparatorsArray.size(); ++i){
            row = lineSeparatorsArray.get(i) / 10;

            //mask this region of interest
            Mat thisLineMask = new Mat(new Size(cols, rows), CvType.CV_8UC1, new Scalar(0));
            Imgproc.rectangle(thisLineMask, new Point(0, prevRow), new Point(cols, row), new Scalar(255), -1);
            
            // trebles_mask_fit_to_line = trebles_mask_shrunk[prev_line:row, :] 
            Mat treblesMaskFitToLine = new Mat();
            Core.bitwise_and(thisLineMask, treblesMaskShrunk, treblesMaskFitToLine);

            // clef becomes tall white rectangle 
            Imgproc.dilate(treblesMaskFitToLine, treblesMaskFitToLine, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, (row - prevRow) * 2)), new Point(-1,-1), 1);
            Core.bitwise_and(thisLineMask, treblesMaskFitToLine, treblesMaskFitToLine);

            // basses_mask_fit_to_line = basses_mask_shrunk[prev_line:row, :] 
            Mat bassesMaskFitToLine = new Mat();
            Core.bitwise_and(thisLineMask, bassesMaskShrunk, bassesMaskFitToLine);

            // clef becomes tall white rectangle
            Imgproc.dilate(bassesMaskFitToLine, bassesMaskFitToLine, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, (row - prevRow) * 2)), new Point(-1,-1), 1);
            Core.bitwise_and(thisLineMask, bassesMaskFitToLine, bassesMaskFitToLine);
        
            musicLinesColored.setTo(new Scalar(0, 255, 0), treblesMaskFitToLine);
            musicLinesColored.setTo(new Scalar(255, 0, 0), bassesMaskFitToLine);
            prevRow = row;
        }
        

        // Step 2: expand the clef areas horizontally

        byte[] prevColor = new byte[3];
        byte[] curr = new byte[3];

        
        for (int i =0 ;i < rows; ++i){
            
            prevColor[0] = 0;
            prevColor[1] = 0;
            prevColor[2] = 0;
            for(int j = 0; j < cols; ++j){
                musicLinesColored.get(i, j, curr);
                if(curr[2] != 0){
                    // is red pixel meaning a line separator. skip this line.
                    break;
                }
                if(curr[0] != 0){
                    //is blue meaning bass clef. 
                    prevColor[0] = curr[0];
                    prevColor[1] = curr[1];
                    prevColor[2] = curr[2];
                }
                else if(curr[1] != 0){
                    //is green meaning treble clef. 
                    prevColor[0] = curr[0];
                    prevColor[1] = curr[1];
                    prevColor[2] = curr[2];
                }
                else{
                    musicLinesColored.put(i, j, prevColor);
                }
            }
        }
        
        Imgproc.resize(musicLinesColored, musicLinesColored, musicLines.size());
        // music_lines = cv.resize(music_lines_shrunk, (music_lines.shape[1], music_lines.shape[0]))

        //red row of pixels between every line
        for (int i =1 ;i < lineSeparatorsArray.size(); ++i)
            musicLinesColored.row(lineSeparatorsArray.get(i)).setTo(new Scalar(0,0,255));


        long endTime = System.currentTimeMillis();
        System.out.println("addClefAreas took " + (endTime - startTime));

        return musicLinesColored;
    }
    // return an array of rows where color changed, including row 0 and row -1
    public static List<Integer> getLineSeparatorsArray(Mat musicLinesMask, Mat musicLines){
        long startTime = System.currentTimeMillis();
        // Imgcodecs.imwrite("./img/musicLinesMask.png", musicLinesMask);
        int rows = musicLinesMask.rows();
        int cols = musicLinesMask.cols();
        // System.out.println("musicLinesMask size : " + rows + ", " + cols);
        List<Integer> arr = new ArrayList<>();
        // arr = np.empty([44], np.uint16)
        arr.add(0);
        byte[] curr = new byte[1];
        byte[] prev = new byte[1];
        prev[0] = (byte)0;
        // arr will contain all the row indices that transition from black to white or white to black
        for(int i = 0; i < rows; ++i){

            musicLinesMask.get(i, cols / 2, curr);
            // System.out.print("curr: " + (int)curr[0]);
            if(curr[0] != prev[0]){
                arr.add(i);
                // System.out.print("curr: " + curr[0]+ "  " + prevColor);
            }
            prev[0] = curr[0];
        }
        arr.add(rows - 1);
        System.out.println("lineSeparatorsArray had length " + arr.size());
    
        // assert arr.size() < 42 : "Image was detected to have over 20 lines! It likely won't convert well, aborting program";
        assert arr.size()% 2 == 0 : "Failed to distinguish the boundaries between lines. Got an odd number of line_separators";

        // row divisions will contain all the row indices that bisect the lines
        List<Integer> rowDivisions = new ArrayList<>();
        // rowDivisions = np.empty([20],np.uint16)
        int row;
        for(int i =0 ; i < arr.size(); i += 2){
            row = arr.get(i) + (arr.get(i + 1) - arr.get(i))/2;
            rowDivisions.add(row);
            // System.out.print(row + "  ");
        }
        assert rowDivisions.size() > 1 : "Failed to idenify boundaries for a single line of music";

        long endTime = System.currentTimeMillis();
        System.out.println("getLineSeparatorsArray took " + (endTime - startTime));
        return rowDivisions;
    }



    //put white between the white staff lines. Used for deciphering a clef's territory
    public static Mat getMusicLinesMask(Mat staffBinary, int noteHeight){
        System.out.println("in getmusiclinesmask");
        long startTime = System.currentTimeMillis();
        // Imgcodecs.imwrite("./img/staffBinary.png", staffBinary);
        Scalar white = new Scalar(255);
        Mat staffBinInv = new Mat();
        Core.bitwise_not(staffBinary, staffBinInv);
        int rows = staffBinInv.rows();
        int cols = staffBinInv.cols();


        //sanitize margins by blacking out the top and bottom 
        byte[] curr = new byte[1];
        int y = 0;
        while(y < staffBinInv.rows()){
            staffBinInv.get(y, cols / 2, curr);
            if(curr[0] != (byte)255) break;
            y++;
        }
        curr[0] = (byte)0;
        
        staffBinInv.put(0, y * rows, curr);
        // staffBinary[0:y, :] = 0
        y = rows - 1;
        while(y < staffBinInv.rows()){
            staffBinInv.get(y, cols / 2, curr);
            if(curr[0] != (byte)255) break;
            y--;
        }
        curr[0] = (byte)0;
        System.out.println("y in getMusicLinesMask: " +y);
        staffBinInv.put(y * rows, rows * cols, curr);
        // staff_binary[y:rows, :] = 0



        Mat musicLinesMask = new Mat();
        staffBinInv.copyTo(musicLinesMask);

        int lineStart = -1;
        int lineEnd = -1;
        int prev = -1;
        int tol = rows / 100;
        //idea: when curr_distance > note_height (+ some tolerance), this is a new line
        for (int i =0; i < rows; ++i){
            // print(staff_binary[i, cols//2])
            staffBinInv.get(i, cols/2, curr);
            if(curr[0] == (byte)255){
                //case 1: top of a new line
            
                if(i - prev > (int)(noteHeight + tol)){
                    
                    // line_end = i
                    if(lineStart > -1 && lineEnd > -1){
                        // print("whited ", line_start, " -> ", line_end)
                        
                        for(int j =lineStart; j < lineEnd; ++j)
                            musicLinesMask.row(j).setTo(white);
                        // musicLinesMask[line_start:line_end, :] = 255
                    }
                    lineStart = i;
                }
                //case 2: not new "line" BEcause we are still somewhere in between some ledger lines
                else
                    lineEnd = i;
                prev = i;
            }
        }
        curr[0] = (byte)255;
        
        for(int i =lineStart; i < lineEnd; ++i)
            musicLinesMask.row(i).setTo(white);
        // musicLinesMask.put(lineStart * rows, lineEnd * rows, curr);
        // music_lines_mask[line_start:line_end, :] = 255
        long endTime = System.currentTimeMillis();
        System.out.println("getMusicLinesMask took " + (endTime - startTime));

        return musicLinesMask;
    }

    
    //TODO: repeat for the smaller, inline clefs present in some pieces, after a downsize
    public static Mat eraseTrebleClefs(Mat img, int noteHeight, String desiredClef, double threshold, Mat newClefs){
        //return a mask of treble clef locations

        long startTime = System.currentTimeMillis();
                
        //read treble clef template and size it appropriately for the user's sheet m;usic
        String trebleTemplatePath = "./templates/trebleclef.jpg";
        Mat trebleClefTemplate = Imgcodecs.imread(trebleTemplatePath, Imgcodecs.IMREAD_GRAYSCALE) ;
        assert !trebleClefTemplate.empty() : "the treble template Mat is empty";
        int upHeight = noteHeight * 7;
        int upWidth = (int)(((double)upHeight /trebleClefTemplate.rows() ) * trebleClefTemplate.cols());
        Mat trebleClefStructuringElement = new Mat(new Size(upWidth ,upHeight),  CvType.CV_8UC1, new Scalar(255)); // used with erosion for mask creation
        Mat trebleClefTemplateResized = new Mat();
        Imgproc.resize(trebleClefTemplate, trebleClefTemplateResized, new Size(upWidth, upHeight), 0, 0,  Imgproc.INTER_LINEAR); //TODO: make this work for inline trebles

        //read bass clef template and size it appropriately for the user's sheet m;usic
        String bassTemplatePath = "./templates/bassclef.jpg";
        Mat bassClefTemplate = Imgcodecs.imread(bassTemplatePath, Imgcodecs.IMREAD_GRAYSCALE) ;
        assert !bassClefTemplate.empty() : "the bass template Mat is empty";
        upHeight = (int)(noteHeight * 3.7);
        upWidth = (int)(((double)upHeight /bassClefTemplate.rows() ) * bassClefTemplate.cols());
        Mat bassClefTemplateResized = new Mat();
        Imgproc.resize(bassClefTemplate, bassClefTemplateResized, new Size(upWidth, upHeight), 0, 0,  Imgproc.INTER_LINEAR);


        // Perform match operations
        Mat trebleMatchesMask = getTemplateMatchesMask(img, trebleClefTemplateResized, threshold);
      
        //convert white pixels to treble sized rectangles
        trebleMatchesMask = coverTemplateMatchWithStructuringElement(trebleMatchesMask, trebleClefStructuringElement);

        //replace the bounding box with the treble inside
        Mat treblesMask = new Mat();
        Core.min(trebleMatchesMask, img, treblesMask);
        Core.bitwise_xor(treblesMask, trebleMatchesMask, treblesMask);
       
        long endTime = System.currentTimeMillis();
        System.out.println("eraseTrebleClefs took " + (endTime - startTime));

        return treblesMask;
        
    }
    //TODO: repeat for the smaller, inline clefs present in some pieces, after a downsize
    public static Mat eraseBassClefs(Mat img, int noteHeight, String desiredClef, double threshold, Mat newClefs){
        //return a mask of bass clef locations

        long startTime = System.currentTimeMillis();
        
        
        //read bass clef template and size it appropriately for the user's sheet m;usic
        String bassTemplatePath = "./templates/bassclef.jpg";
        Mat bassClefTemplate = Imgcodecs.imread(bassTemplatePath, Imgcodecs.IMREAD_GRAYSCALE) ;
        assert !bassClefTemplate.empty() : "the bass template Mat is empty";
        int upHeight = (int)(noteHeight * 3.7);
        int upWidth = (int)(bassClefTemplate.cols() * upHeight /bassClefTemplate.rows() );
        Mat bassClefStructuringElement = new Mat(new Size(upWidth ,upHeight),  CvType.CV_8UC1, new Scalar(255)); // used with erosion for mask creation
        Mat bassClefTemplateResized = new Mat();
        Imgproc.resize(bassClefTemplate, bassClefTemplateResized, new Size(upWidth, upHeight), 0, 0,  Imgproc.INTER_LINEAR);

        //read bass clef template and size it appropriately for the user's sheet m;usic
        String trebleTemplatePath = "./templates/trebleclef.jpg";
        Mat trebleClefTemplate = Imgcodecs.imread(trebleTemplatePath, Imgcodecs.IMREAD_GRAYSCALE) ;
        assert !trebleClefTemplate.empty() : "the treble template Mat is empty";
        upHeight = noteHeight * 7;
        upWidth = (int)( trebleClefTemplate.cols() * upHeight / trebleClefTemplate.rows() );
        Mat trebleClefTemplateResized = new Mat();
        Imgproc.resize(trebleClefTemplate, trebleClefTemplateResized, new Size(upWidth, upHeight), 0, 0,  Imgproc.INTER_LINEAR);

        
        // Perform match operations
        Mat bassMatchesMask = getTemplateMatchesMask(img, bassClefTemplateResized, threshold);
      
        //convert white pixels to treble sized rectangles
        bassMatchesMask = coverTemplateMatchWithStructuringElement(bassMatchesMask, bassClefStructuringElement);


        //replace the bounding box with the bass inside
        Mat bassesMask = new Mat();
        Core.min(bassMatchesMask, img, bassesMask);
        Core.bitwise_xor(bassesMask, bassMatchesMask, bassesMask);

        long endTime = System.currentTimeMillis();
        System.out.println("eraseBassClefs " + (endTime - startTime));

        return bassesMask;
    }

    public static int getAverageNoteHeight(Mat staffBinary, List<Integer> gapHeights){
        long startTime = System.currentTimeMillis();
        assert staffBinary.channels() == 1 : "getAverageNoteHeight got a multichannel image";

        
        int rows = staffBinary.rows();
        int cols = staffBinary.cols();

        //TODO: consider optimality 
        int tol = rows / 50 ;
        List<Integer> lineStarts = new ArrayList<>(); //list of row heights for each staff line (only their topmost pixel)
        
        byte[] curr = new byte[1];
        boolean prevIsZero = false;
        for(int i =0 ; i < rows; ++i){
            staffBinary.get(i,cols/2 +1, curr);
            if (curr[0] == 0 && !prevIsZero) //start of black zone
                lineStarts.add(i);
            prevIsZero = curr[0] == 0 ? true : false;
        }
        int totalSum = 0;
        int divisor = 0;
        int newGap;
        for(int i =1; i < lineStarts.size(); ++i){
            newGap = lineStarts.get(i) - lineStarts.get(i-1);
            if(newGap < tol){
                totalSum += newGap;
                divisor += 1;
            }
        }
        int average = totalSum  / divisor;
        long endTime = System.currentTimeMillis();
        System.out.println("getAverageNoteHeight took " + (endTime - startTime));
        return average;
    }

    public static Mat isolateStaff(Mat img, int lineWidth){
        // HighGui.imshow("img", img);
        // HighGui.waitKey(0);
        //TODO: pick the intensity just before the right peaks of histr plot
        int staffUpperThreshold = 235; //takefive.png
        // staff_thresh_intensity = 220 //obsessed.png (musescore)

        Mat arbitraryThreshold = new Mat();
        double ret = Imgproc.threshold(img, arbitraryThreshold, staffUpperThreshold,255,Imgproc.THRESH_BINARY); //we normally use otsu thresholding, but use this so fewer grays will be turned to white
        // HighGui.imshow("arbitraryThreshold", arbitraryThreshold);
        // HighGui.waitKey(0);
        int width = img.cols();
        //Step 1 : get just the staff
        Mat selectStaffKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,1));  //kernel for eroding thin white lines - first, will be used to get just the staff

        Mat staffShortened = new Mat();
        Core.bitwise_not(arbitraryThreshold, arbitraryThreshold);
        Imgproc.erode(arbitraryThreshold, staffShortened, selectStaffKernel, new Point(-1,-1), width / 6); // Note: image will be blank if iterations is too high. Must be less than staff length divided by (I believe) 3       Perfect before our Hough transform
        Mat staffLengthened = new Mat();
        Imgproc.dilate(staffShortened, staffLengthened, selectStaffKernel, new Point(-1,-1), width / 2 ); // Note: can be done infinitely without affecting future steps; we expect the staff to be too long anyway after this step

        // HighGui.imshow("staffShortened", staffShortened);
        // HighGui.waitKey(0);
        // HighGui.imshow("staffLengthened", staffLengthened);
        // HighGui.waitKey(0);
        Core.bitwise_not(staffLengthened, staffLengthened);
        return staffLengthened;
    }
    //simply returns 100 for now, works well for musescore
    //TODO: decide a threshold value based on the histograms of the staff. idea: return the bin just before the right side climb.
    public static int estimateStaffUpperThreshold(Mat grayscaleStaff){
        return 100;
    }

    public static List<Integer> segmentStaffs(Mat img){
        long startTime = System.currentTimeMillis();
        Imgproc.threshold(img, img, 254, 255, Imgproc.THRESH_BINARY);
        int rows = img.rows();
        int cols = img.cols();
        // print('enlarged shape: ', img.shape)
        
        Mat halfWidthStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(cols/2,1));
        Mat fullWidthStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(cols,1));
        Mat staffGaps = new Mat();
        // System.out.println("segmentStaff channels is " + img.channels() + " and type is " + CvType.typeToString(img.type()));
        // //ideally would erode only at this mask :(
        // Mat erodeMask = new Mat(new Size(rows, cols), CvType.CV_8UC1, new Scalar(0));
        // erodeMask.col(cols/2).setTo(new Scalar(255));
        
        // long startErode = System.currentTimeMillis();
        // Imgproc.erode(img, staffGaps, halfWidthStructure, new Point(-1,-1), 5); // TODO: for staff finding errrors increase the iterations
        // Imgproc.erode(img, staffGaps, fullWidthStructure, new Point(-1,-1), 1); //too slow
        // long endErode = System.currentTimeMillis();
        // HighGui.imshow("staffGaps", staffGaps);
        // HighGui.waitKey(0);
        // System.out.println("staff erosion took " + (endErode - startErode));
        // System.out.println("img type was" + (CvType.typeToString(img.type())));

        
        List<Integer> gapHeights = new ArrayList<>();// will store middle (height) of each gap
        gapHeights.add(0);

        int tol = rows / 10;    // tolerance to prevent adding gaps that aren't tall enough
        byte[] curr = new byte[1]; //the current pixel intensity
        // System.out.println("staffgap chanel count: " + staffGaps.channels());
        // System.out.println("staffgap type: " + CvType.typeToString(staffGaps.type()));

        boolean isWhite;
        int prevHeight = 0;
        int count;
        int max = 0;
        for (int i = 0; i < rows; i++){ 
            isWhite = true;
            count = 0;

            for (int j = 0; j < rows; j++){ 
                if(count > img.width() / 100) break;
                img.get(i, j, curr);
                // System.out.print( (int)curr[0] + " " + (curr[0] == 0));
                if(curr[0] == 0 ){
                    isWhite = false;
                    count++;
                }
                max = Math.max(max, j);
            }
            if(isWhite == true && i - prevHeight > tol) {
                gapHeights.add(i);
                prevHeight = i;
            }
        }
        // System.out.println("longest white stretch was " + max);
        
        assert gapHeights.size() > 1 : "Error finding any gaps between the staff. Proceed if image is one line";

        long endTime = System.currentTimeMillis();
        System.out.println("segmentStaffs took " + (endTime - startTime));
        return gapHeights;

    }
    //resize img to at least width x height, keeping its aspect ratio
    public static Mat getEnlarged(Mat img, int width,  int height){
        int upHeight = -1;
        int upWidth = -1;

        if(img.rows() < height){
            upHeight = 2200;
            upWidth = img.cols() * (upHeight / img.rows());
        }
        if(upWidth < width){
            int temp = upWidth > 0 ? upWidth : img.cols();
            upWidth = width;
            upHeight = (upWidth / temp * upHeight);
        }
        
        if(upHeight == -1)  //img already larger than the width, height
            return img;

        Mat imgEnlarged = new Mat();;
        Imgproc.resize(img, imgEnlarged, new Size(upWidth, upHeight), 0, 0,  Imgproc.INTER_LINEAR);
        return imgEnlarged;
    }

    public static Mat convertClef(Mat img, String desiredClef){
        assert !img.empty() :  "convert_clef's img was empty, couldn't convert";
        assert desiredClef.equals( "bass") || desiredClef.equals("treble") :  "desired_clef should be bass or treble, got " + desiredClef;

        
        if(img.channels()>2)
            Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        boolean isPiano = false;    //if braces are template matched, this becomes True 

        /*
        BEGIN ENLARGE IMAGE FOR ACCURATE TEMPLATE MATCHES
        */

        System.out.println("Pre-processing started");
        System.out.println("img size: " +  img.size());

        img = getEnlarged(img, 1700, 2200);
        System.out.println("imgEnlarged size: " +  img.size());


        /* 
        FINISH ENLARGE IMAGE FOR ACCURATE TEMPLATE MATCH
        */

        //declaring useful modified Mats early.
        int rows = img.rows();
        int cols = img.cols();
        Mat grayscaleInv = new Mat();
        Mat imgBinary = new Mat();
        Mat imgBinaryInv = new Mat();
        Core.bitwise_not(img, grayscaleInv);
        Imgproc.adaptiveThreshold(grayscaleInv,  imgBinaryInv, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, -2);
        Core.bitwise_not(imgBinaryInv, imgBinary);



        /*
        ESTIMATE THE STAFF LINES AND REMOVE THEM
        */


        List<Integer> gapHeights = segmentStaffs(imgBinary);
        System.out.println("gapHeights:");
        for (int i =0; i < gapHeights.size(); ++i){
            System.out.println(gapHeights.get(i));
        }
        assert gapHeights.size() > 1 : "found no gaps when looking for staff";

        int lineWidth = gapHeights.get(1) - gapHeights.get(0); //size between gaps is the staff width  (line width)
        
        Mat completeStaffBinary = isolateStaff(img.clone(), lineWidth); //modifies grayscale!!!!

        Mat staffMask = new Mat();
        Core.bitwise_not(completeStaffBinary, staffMask);

        Mat grayscaleStaff = new Mat();
        img.copyTo(grayscaleStaff, staffMask); //TODO: Check this

        int staffUpperThreshold = estimateStaffUpperThreshold(grayscaleStaff);





        //staff_threshed contains only the dark symbol overlaps
        Mat staffThreshed = new Mat();
        Imgproc.threshold(grayscaleStaff, staffThreshed, staffUpperThreshold, 255, Imgproc.THRESH_BINARY);

        //don't just erase the entire staff before finding symbols. Erase the staff where it isnt as dark as the symbols! Using staff_threshed
        Mat staffNoOverlaps = new Mat();
        grayscaleStaff.copyTo(staffNoOverlaps);

        Mat staffThreshedIsZeroMask = new Mat();
        Imgproc.threshold(staffThreshed, staffThreshedIsZeroMask, 0, 255, Imgproc.THRESH_BINARY_INV);
        staffNoOverlaps.setTo(new Scalar(255), staffThreshedIsZeroMask);
        
        Mat staffRemoved = new Mat();
        img.copyTo(staffRemoved);
        Mat staffNoOverlapsNotWhiteMask = new Mat();
        Imgproc.threshold(staffNoOverlaps, staffNoOverlapsNotWhiteMask, 254, 255, Imgproc.THRESH_BINARY_INV);
        // staffRemoved[staffNoOverlaps<255] = 255;
        staffRemoved.setTo(new Scalar(255), staffNoOverlapsNotWhiteMask);
        
        // return staffRemoved;
        // crucial for template sizing
        int noteHeight = getAverageNoteHeight(completeStaffBinary, gapHeights); //important for resizing our templates 
        
        System.out.println("average note height: " +  noteHeight);



        /*
        END ESTIMATE STAFF
        */



        /*
        BEGIN TEMPLATE MATCHING
        */


        //we will template match staff_removed for static symbols which won't be translated vertically 

        //stores all symbols that we are going to erase through pattern matching
        //TODO: just make this a binary mask and get real values later
        Mat staticSymbolsMask = new Mat(img.size(), img.type(), new Scalar(255));
        Mat staffRemovedCopy = new Mat();
        staffRemoved.copyTo(staffRemovedCopy);

        //: dilate staff_removed (inverted) to recover staff overlap
        Mat staffRemovedBin = new Mat();
        Imgproc.threshold(staffRemoved, staffRemovedBin,200,255,Imgproc.THRESH_BINARY); // wider range of grays will be kept as black than in adaptive thresholding


        Mat staffRemovedBinDilated = new Mat();
        Mat staffRemovedBinInv = new Mat();
        Core.bitwise_not(staffRemovedBin, staffRemovedBinInv);
        Imgproc.dilate(staffRemovedBinInv, staffRemovedBinDilated, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 3)), new Point(-1,-1),1);
        Core.bitwise_not(staffRemovedBinDilated, staffRemovedBinDilated);




        Mat lightSymbolOverlapsMask = new Mat();
        staffRemovedBin.copyTo(lightSymbolOverlapsMask);

        Mat diffMask = new Mat();
        Core.compare(staffRemovedBinDilated, staffRemovedBin, diffMask, Core.CMP_NE);
        lightSymbolOverlapsMask.setTo(new Scalar(255), diffMask);
        Imgproc.threshold(lightSymbolOverlapsMask, lightSymbolOverlapsMask, 1, 255, Imgproc.THRESH_BINARY_INV);
        img.setTo(new Scalar(255), lightSymbolOverlapsMask);
     

        //get combined ink
        // Core.min(img, staffRemoved, staffRemoved); // can skip now????


        Mat newClefs = new Mat(img.rows(), img.cols(), img.type(), new Scalar((byte)255));

        Mat treblesMask = eraseTrebleClefs(staffRemoved, noteHeight,  desiredClef, 0.55, newClefs);
        Mat trebleClefsMask = eraseTrebleClefs(staffRemoved, noteHeight,  desiredClef, 0.55, newClefs);
        Mat bassClefsMask = eraseBassClefs(staffRemoved, noteHeight, desiredClef, 0.55, newClefs);
        
        

        Imgcodecs.imwrite("./img/trebleMask.png",  trebleClefsMask);
        Imgcodecs.imwrite("./img/bassMask.png",  bassClefsMask);
        // cv.imwrite(intermediate_output_file_name,new_clefs)
        if(desiredClef.equals("treble"))
            Core.max(staticSymbolsMask, trebleClefsMask, staticSymbolsMask);
        else if(desiredClef.equals( "bass"))
            Core.max(staticSymbolsMask, bassClefsMask, staticSymbolsMask);

        // System.out.println("completeStaffBinary shape: " + completeStaffBinary.rows() + ", " + completeStaffBinary.cols());
        
        Mat musicLinesMask = getMusicLinesMask(completeStaffBinary, noteHeight);
        
        Imgcodecs.imwrite("./img/musicLinesMask.png", musicLinesMask);

        Mat musicLines = new Mat();
        Imgproc.cvtColor(musicLinesMask, musicLines, Imgproc.COLOR_GRAY2RGB);

        // //draw red lines to divide the staff lines, return their row indices
        List<Integer> lineSeparatorsArray = getLineSeparatorsArray(musicLinesMask, musicLines);
        

        musicLines = addClefAreas(lineSeparatorsArray, musicLines, trebleClefsMask, bassClefsMask);
        

        // //quarter, eighth, sixteenth rests
        Mat shortRestsMask = eraseShortRests(staffRemoved, noteHeight, 0.63);

        









        // static_symbols_mask = cv.bitwise_or(static_symbols_mask, short_rests_mask)

        // words_mask = erase_words_no_resize(img=staff_removed, threshold = 92, note_height=note_height) #pytesseract wants thresh as %
        // static_symbols_mask = cv.bitwise_or(static_symbols_mask, words_mask)

        // braces_mask = erase_braces(img=staff_removed, threshold=0.55, note_height=note_height)
        // if(braces_mask.__contains__(255))
        //     is_piano = True
        // static_symbols_mask = cv.bitwise_or(static_symbols_mask, braces_mask)

        // time_signatures_mask = erase_time_signatures(img=staff_removed, note_height=note_height, threshold = 0.73)
        // static_symbols_mask = cv.bitwise_or(static_symbols_mask, time_signatures_mask)


        // //get make sure the next template matches are not touching notes
        // noteheads_mask = get_notehead_mask(img=staff_removed, note_height=note_height)
        
        // double_bar_lines_mask = erase_double_bar_lines(img=staff_removed,note_height=note_height,threshold=0.8, is_piano=is_piano)
        // static_symbols_mask = cv.bitwise_or(static_symbols_mask, double_bar_lines_mask)

        // //TODO: erase only if passing template match is perfect height
        // bar_lines_mask = erase_bar_lines(img = staff_removed, noteheads_mask = noteheads_mask,  threshold = 0.55 , note_height = note_height, is_piano = is_piano)
        // static_symbols_mask = cv.bitwise_or(static_symbols_mask, bar_lines_mask)
        // // cv.imwrite(intermediate_output_file_name, static_symbols_mask)
        // // print("made it out of static_symbol")
        // // erase_long_rests(img = staff_removed, noteheads_mask=noteheads_mask, threshold = 0.5, note_height=note_height)




        /*
        END TEMPLATE MATCHING
        */






        // Core.max(staffRemoved, trebleClefsMask, staffRemoved);
        // return shortRestsMask;
        return musicLines;
        // return trebleClefsMask;
        // return bassClefsMask;
        // return newClefs;
        // return shortRestsMask;
        // return staffRemoved;
    }






	public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        
        String desiredClef = "bass";

        String inputFileName = "./img/obsessed.png";
        // String inputFileName = "./img/clair.png";
        String outputFileName = "./img/output.png";
        // File inputFile = new File(inputFileName);
        // assert inputFile.exists() : inputFileName + " not found, aborting";
        // File outputFile = new File(outputFileName);
        

        //validate input_file_name path
        long startRead = System.currentTimeMillis();
        Mat img = Imgcodecs.imread(inputFileName, Imgcodecs.IMREAD_GRAYSCALE);
        long endRead = System.currentTimeMillis();
        System.out.println("read time: " + (endRead - startRead));
        
        assert !img.empty() : "the input image became an empty Mat";
        
        Mat transposedImage = convertClef(img, desiredClef);
        

        //validate output_file_name path
        // long startWrite = System.currentTimeMillis();
        boolean write_status = Imgcodecs.imwrite(outputFileName,  transposedImage);
        // long endWrite = System.currentTimeMillis();
        // System.out.println("write time: " + (endWrite - startWrite));
        
        assert write_status : "issue during imwrite()";
        long endTime = System.currentTimeMillis();
        System.out.println("total time: " + (endTime - startTime));
	}
}