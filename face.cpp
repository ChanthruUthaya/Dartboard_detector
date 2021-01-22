/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>

using namespace std;
using namespace cv;

/** Function Headers */
std::vector<Rect> detectAndDisplay(Mat frame);
float iou(Rect face, Rect face2);
void scores(std::vector<Rect> faces, std::vector<Rect> groundtruth);
float tpr(std::vector<Rect>& groundtruth, int& tp);
float f1Score(std::vector<Rect>& faces, std::vector<Rect>& groundtruth, int& tp);
Mat thresholding(cv::Mat& grad, float thresh);
vector<tuple<int, int, int>> hough(cv::Mat& thresholdimage, cv::Mat& orienationimage, int x0, int y0, int x1, int y1, int houghthresh, int radthresh, int minr, int maxr);
std::pair<int, int> circleCenter(int radius, std::pair<int, int> pos, double grad);
Mat sobeldx(cv::Mat& image);
vector<tuple<int, int, int>> sobel(const char* name, Mat& imname);
vector<pair<int, int>> sobel2(Mat& frame, int x0, int y0, int x1, int y1, tuple<Mat, Mat, Mat, Mat> lineimages);
Mat sobeldy(cv::Mat& image);
cv::Mat sobelmag(cv::Mat& dx, cv::Mat& dy);
int*** malloc3dArray(int dim1, int dim2, int dim3);
pair<Mat, Mat> sobeldir(cv::Mat& dx, cv::Mat& dy);
tuple<Mat, Mat, Mat, Mat> sobelReturn(Mat& image);
vector<pair<int, int>> houghTransformLines(Mat treshholded, Mat& dir, Mat& frame, int x0, int y0, int x1, int y1, int houghlinethresh, int lineintersectionthresh);
int** malloc2DArray(int x, int y);
int diagonalOfImage(int x, int);
Mat tresholdHough2D(Mat normalizedHough, int rhoRange, int thetaRange, int threshold);
pair<vector<int>, vector<pair<int, int>>> boardcenters(vector<tuple<int, int, int>> circles);
vector<tuple<int, int, int>> boarddetection(vector<pair<int, int>> centers, vector<int> r, vector<pair<int, int>> intersections, float dist);
vector<pair<int, int>> findLineIntersections(std::vector<tuple<int, int>> lines, Mat treshholded, int lineintersectionthresh, int x0, int y0, int x1, int y1);
Mat thresholdLineIntersections(Mat normalizedLineIntersections, int threshold);
vector<pair<int, int>> intersectionCenters(vector<pair<int, int>> lines);
vector<Rect> makeRects(vector<tuple<int, int, int>> boards);


/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main(int argc, const char** argv)
{
	// 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);


	// 2. Load the Strong Classifier in a structure called `Cascade'
	if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	std::vector<Rect> violajones = detectAndDisplay(frame);

	if (argc == 4 && *argv[3] == '1') {

		std::ifstream infile(argv[2]);
		std::vector<Rect> groundtruth;
		int a, b, c, d;
		while (infile >> a >> b >> c >> d) {
			Rect rect = Rect(a, b, c, d);
			groundtruth.push_back(rect);
		}
		std::cout << groundtruth.size() << std::endl;
		scores(violajones, groundtruth);
	}
	else if (argc == 2) {
		vector<tuple<int, int, int>> centersFromEdgeDetect = sobel(argv[1], frame);
		Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
		Mat blur = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
		GaussianBlur(blur, blur, Size(9, 9), 2, 2);
		Mat padded;
		Mat blurpadded;
		cv::copyMakeBorder(image, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
		cv::copyMakeBorder(blur, blurpadded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
		tuple<Mat, Mat, Mat, Mat> lineimages = sobelReturn(blurpadded);
		tuple<Mat, Mat, Mat, Mat> circleimages = sobelReturn(image);
		vector<pair<int, int>> allcenters;
		for (int i = 0; i < violajones.size(); i++) {
			int x0 = violajones[i].x;
			int y0 = violajones[i].y;
			int x1 = x0 + violajones[i].width;
			int y1 = y0 + violajones[i].height;
			vector<pair<int, int>> centersfromVJ = sobel2(frame, x0, y0, x1, y1, lineimages);
			for (int j = 0; j < centersfromVJ.size(); j++) {
				allcenters.push_back(centersfromVJ[j]);
			}
		}
		vector<int> rs;
		vector<pair<int, int>> cps;
		for (int j = 0; j < centersFromEdgeDetect.size(); j++) {
			rs.push_back(get<2>(centersFromEdgeDetect[j]));
			cps.push_back(make_pair(get<0>(centersFromEdgeDetect[j]), get<1>(centersFromEdgeDetect[j])));
		}
		vector<tuple<int, int, int>> detectedpoints = boarddetection(cps, rs, allcenters, 20);
		vector<Rect> detected = makeRects(detectedpoints);
		vector<Rect> toPlot;
		for (int j = 0; j < detected.size(); j++) {
			int xc = detected[j].x;
			int yc = detected[j].y;
			int w = detected[j].width;
			int h = detected[j].height;
			bool found = false;
			for (int i = 0; i < toPlot.size(); i++) {
				int x1 = toPlot[i].x;
				int y1 = toPlot[i].y;
				int w1 = toPlot[i].width;
				int h1 = toPlot[i].height;
				if (xc == x1 && yc == y1 && w == w1 && h == h1) {
					found = true;
				}
			}
			if (!found) {
				toPlot.push_back(detected[j]);
			}
		}
		cout << toPlot.size() << endl;
		for (int j = 0; j < toPlot.size(); j++) {
			rectangle(frame, toPlot[j], Scalar(0, 255, 0), 3);
		}
	}
	else if (argc == 4 && *argv[3] == '3') {
		std::vector<Rect> faces = detectAndDisplay(frame);
		std::ifstream infile(argv[2]);
		std::vector<Rect> groundtruth;
		int a, b, c, d;
		while (infile >> a >> b >> c >> d) {
			Rect rect = Rect(a, b, c, d);
			rectangle(frame, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), Scalar(0, 0, 255), 2);
			groundtruth.push_back(rect);
		}
		vector<tuple<int, int, int>> centersFromEdgeDetect = sobel(argv[1], frame);
		Mat blur = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
		GaussianBlur(blur, blur, Size(9, 9), 2, 2);
		Mat blurpadded;
		cv::copyMakeBorder(blur, blurpadded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
		tuple<Mat, Mat, Mat, Mat> lineimages = sobelReturn(blurpadded);
		vector<pair<int, int>> allcenters;
		for (int i = 0; i < violajones.size(); i++) {
			int x0 = violajones[i].x;
			int y0 = violajones[i].y;
			int x1 = x0 + violajones[i].width;
			int y1 = y0 + violajones[i].height;
			vector<pair<int, int>> centersfromVJ = sobel2(frame, x0, y0, x1, y1, lineimages);
			for (int j = 0; j < centersfromVJ.size(); j++) {
				allcenters.push_back(centersfromVJ[j]);
			}
		}
		vector<int> rs;
		vector<pair<int, int>> cps;
		for (int j = 0; j < centersFromEdgeDetect.size(); j++) {
			rs.push_back(get<2>(centersFromEdgeDetect[j]));
			cps.push_back(make_pair(get<0>(centersFromEdgeDetect[j]), get<1>(centersFromEdgeDetect[j])));
		}
		vector<tuple<int, int, int>> detectedpoints = boarddetection(cps,rs,allcenters,20);
		vector<Rect> detected = makeRects(detectedpoints);
		vector<Rect> toPlot;
		for (int j = 0; j < detected.size(); j++) {
			int xc = detected[j].x;
			int yc = detected[j].y;
			int w = detected[j].width;
			int h = detected[j].height;
			bool found = false;
			for (int i = 0; i < toPlot.size(); i++) {
				int x1 = toPlot[i].x;
				int y1 = toPlot[i].y;
				int w1 = toPlot[i].width;
				int h1 = toPlot[i].height;
				if (xc == x1 && yc == y1 && w == w1 && h == h1) {
					found = true;
				}
			}
			if (!found) {
				toPlot.push_back(detected[j]);
			}
		}
		cout << toPlot.size() << endl;
		for (int j = 0; j < toPlot.size(); j++) {
			rectangle(frame, toPlot[j], Scalar(0, 255, 0), 3);
		}
		scores(toPlot, groundtruth);
	}

	// 4. Save Result Image
	imwrite("detected.jpg", frame);
	imshow("detected", frame);
	waitKey(0);
	cvDestroyAllWindows();

	return 0;
}

vector<tuple<int, int, int>> sobel(const char* name, Mat& imname) {
	Mat image = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	Mat blur = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	Mat frame = imread(name, CV_LOAD_IMAGE_COLOR);
	GaussianBlur(blur, blur, Size(9, 9), 2, 2);
	Mat padded;
	Mat blurpadded;
	cv::copyMakeBorder(blur, blurpadded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
	cv::copyMakeBorder(image, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
	Mat dx = sobeldx(padded);
	Mat dy = sobeldy(padded);
	Mat grad = sobelmag(dx, dy);
	pair<Mat, Mat> dirp = sobeldir(dx, dy);
	Mat dir = dirp.first;
	Mat outdir = dirp.second;
	tuple<Mat, Mat, Mat, Mat> lineimages = sobelReturn(blur);
	Mat threshline = thresholding(get<2>(lineimages), 130);
	Mat threshcircle = thresholding(grad, 150);
	int x0 = 0;
	int y0 = 0;
	int x1 = threshline.cols-1;
	int y1 = threshline.rows-1;
	imwrite("sobelout/dx.jpg", dx);
	imwrite("sobelout/dy.jpg", dy);
	imwrite("sobelout/maggrad.jpg", grad);
	imwrite("sobelout/dirgrad.jpg", outdir);
	imwrite("sobelout/thresh.jpg", threshcircle);
	vector<tuple<int, int, int>> circles = hough(threshcircle, dir,x0, y0,x1,y1, 130, 32 , 20, 150);
	pair<vector<int>, vector<pair<int, int>>> bc = boardcenters(circles);
	vector<pair<int, int>> p = bc.second;
	vector<int> rs = bc.first;
	vector<tuple<int, int, int>> cp;
	for (int i = 0; i < p.size(); i++) {
		cp.push_back(make_tuple(p[i].first, p[i].second, rs[i]));
	}
	imwrite("sobelout/detected.jpg", frame);
	return cp;
}

vector<pair<int, int>> sobel2(Mat& frame, int x0, int y0, int x1, int y1, tuple<Mat, Mat, Mat, Mat> lineimages) {
	Mat threshline = thresholding(get<2>(lineimages), 130);
	vector<pair<int, int>> intersections = houghTransformLines(threshline, get<3>(lineimages), frame, x0, y0, x1, y1,90, 190);
	return intersections;
}


vector<tuple<int, int, int>> boarddetection(vector<pair<int, int>> centers, vector<int> r, vector<pair<int, int>> intersections, float dist) {
	vector<tuple<int, int, int>> boards;
	for (int i = 0; i < centers.size(); i++) {
		int cx = centers[i].first;
		int cy = centers[i].second;
		for (int j = 0; j < intersections.size(); j++) {
			int ix = intersections[j].first;
			int iy = intersections[j].second;
			float d = sqrt((ix - cx) * (ix - cx) + (iy - cy) * (iy - cy));
			if (d < dist) {
				boards.push_back(make_tuple(cx, cy, r[i]));
			}
		}
	}
	return boards;
}

vector<Rect> makeRects(vector<tuple<int, int, int>> boards) {
	vector<Rect> rectangles;
	for (int k = 0; k < boards.size(); k++) {
		int cx = get<0>(boards[k]);
		int cy = get<1>(boards[k]);
		int r = get<2>(boards[k]);
		int rectx = cx - r;
		int recty = cy - r;
		Rect rect = Rect(rectx, recty, 2 * r, 2 * r);
		rectangles.push_back(rect);
	}
	return rectangles;
}

pair<vector<int>, vector<pair<int, int>>> boardcenters(vector<tuple<int, int, int>> circles) {
	int centers = 0;
	vector<int> maxr;
	vector<pair<int, int>> center_means;
	for (int i = 0; i < circles.size(); i++) {


		int x = get<0>(circles[i]);
		int y = get<1>(circles[i]);
		int r = get<2>(circles[i]);

		if (i == 0) {
			centers++;
			center_means.push_back(make_pair(x, y));
			maxr.push_back(r);
		}
		else {
			float min = 10000.0;
			pair<int, int> p = make_pair(0, 0);
			for (int j = 0; j < center_means.size(); j++) {
				int mx = center_means[j].first;
				int my = center_means[j].second;
				float dist = sqrt((mx - x) * (mx - x) + (my - y) * (my - y));
				if (dist < min) {
					min = dist;
					p = make_pair(mx, my);
				}
			}
			if (min > 30.0) {
				centers++;
				center_means.push_back(make_pair(x, y));
				maxr.push_back(r);
			}
			else {
				int index = 0;
				for (int j = 0; j < center_means.size(); j++) {
					if (center_means[j] == p) {
						index = j;
					}
				}
				if (maxr[index] <= r) {
					maxr[index] = r;
				}
				center_means[index].first = floor((center_means[index].first + x) / 2);
				center_means[index].second = floor((center_means[index].second + y) / 2);
			}
		}
	}
	return make_pair(maxr, center_means);
}

void scores(std::vector<Rect> faces, std::vector<Rect> groundtruth) {
	double treshold = 0.34;
	int tp = 0;
	for (int i = 0; i < groundtruth.size(); i++) {
		float max = 0.0;
		for (int j = 0; j < faces.size(); j++) {
			float iouval = iou(faces[j], groundtruth[i]);
			std::cout << iouval << std::endl;
			if (iouval > treshold) {
				
				tp++;
			}
		}
	}
	tpr(groundtruth, tp);
	f1Score(faces, groundtruth, tp);
}

float tpr(std::vector<Rect>& groundtruth, int& tp) {
	float tpr = -1;
	if (groundtruth.size() == 0) {
		std::cout << "\ntpr: div by 0, no groundtruth faces" << std::endl;
	}
	else {
		tpr = (float)tp / groundtruth.size();
		std::cout << "\ntpr: " << tpr << std::endl;
	}
	return  tpr;
}

float f1Score(std::vector<Rect>& faces, std::vector<Rect>& groundtruth, int& tp) {
	float f1 = -1;
	int fp = faces.size() - tp;
	int fn = groundtruth.size() - tp;
	if ((tp + fp) == 0) {
		std::cout << "\n\nno face detected" << std::endl;
		return f1;
	}
	if ((tp + fn) == 0) {
		std::cout << "\n\nno actual face in the image" << std::endl;
		return f1;
	}
	float precision = (float)tp / (tp + fp);
	float recall = (float)tp / (tp + fn);
	if ((precision + recall) == 0) {
		std::cout << "\n\nprecision + recall == 0" << std::endl;
		return f1;
	}
	f1 = 2 * ((precision * recall) / (precision + recall));
	std::cout << "\ntrue positives: " << tp << "\nfalse positives: " << fp << "\nfalse negatives: " <<
		fn << "\n\nf1: " << f1 << std::endl;
	return f1;
}

float iou(Rect face, Rect face2) {
	int area = face.height * face.width;
	int area2 = face2.height * face2.width;
	std::pair<int, int> interpoint1;
	std::pair<int, int> interpoint2;
	interpoint1.first = max(face.x, face2.x);
	interpoint1.second = max(face.y, face2.y);
	interpoint2.first = min(face.x + face.width, face2.x + face2.width);
	interpoint2.second = min(face.y + face.height, face2.y + face2.height);
	int interarea = max(0, interpoint2.first - interpoint1.first) * max(0, interpoint2.second - interpoint1.second);
	int unionarea = area + area2 - interarea;
	float iouval = (float)interarea / unionarea;
	return iouval;
}

Mat tresholdHough2D(Mat normalizedHough, int rhoRange, int thetaRange, int threshold) {
	Mat treshedHough(rhoRange, thetaRange, DataType<int>::type, Scalar(0));

	for (int i = 0; i < rhoRange; i++) {
		for (int j = 0; j < thetaRange; j++)
			if (normalizedHough.at<int>(i, j) > threshold) {
				treshedHough.at<int>(i, j) = 255;
			}
			else treshedHough.at<int>(i, j) = 0;
	}
	return treshedHough;
}

int** malloc2DArray(int x, int y) {
	int** array2D = (int**)malloc(x * sizeof(int*));
	for (int i = 0; i < x; i++) {
		array2D[i] = (int*)malloc(y * sizeof(int));
	}
	for (int i = 0; i < x; i++)
		for (int j = 0; j < y; j++)
			array2D[i][j] = 0;
	return array2D;
}

int diagonalOfImage(int x, int y) {
	return ceil(sqrt(x * x + y * y));
}

Mat thresholding(cv::Mat& grad, float thresh) {
	Mat thresholded(grad.rows, grad.cols, DataType<int>::type, Scalar(0));
	for (int y = 0; y < grad.rows; y++) {
		for (int x = 0; x < grad.cols; x++) {
			if (grad.at<float>(y, x) < thresh) {
				thresholded.at<int>(y, x) = 0;
			}
			else {
				thresholded.at<int>(y, x) = 255;
			}
		}
	}
	return thresholded;
}

vector<tuple<int, int, int>> hough(cv::Mat& thresholdimage, cv::Mat& orienationimage, int x0, int y0, int x1, int y1, int houghthresh, int radthresh, int minr, int maxr) {

	int offset = minr;
	int dim1 = x1 - x0 + 1;
	int dim2 = y1 - y0 +1;
	int dim3 = maxr - minr + 1;

	int*** hougharr = malloc3dArray(dim1, dim2, dim3);

	for (int y = y0; y < y1+1; y++) {
		for (int x = x0; x < x1+1; x++) {
			if (thresholdimage.at<int>(y, x) == 255) {
				for (int l = -1; l < 2; l++) {
					for (int r = minr; r < maxr + 1; r++) {
						double grad = orienationimage.at<double>(y, x);
						pair<int, int> p = circleCenter(r, std::make_pair(x, y), grad + l * 0.1 * M_PI / 180);
						int xc = p.first;
						int yc = p.second;
						if (xc >= x0 && xc <= x1 && yc >= y0 && yc <= y1) {
							hougharr[xc-x0][yc-y0][r - minr] += 1;
						}
					}
				}
			}
		}
	}

	Mat temp(dim2, dim1, DataType<int>::type, Scalar(0));
	Mat output(dim2, dim1, DataType<int>::type, Scalar(0));
	for (int y = 0; y < dim2; y++) {
		for (int x = 0; x < dim1; x++) {
			int result = 0;
			for (int r = 0; r < dim3; r++) {
				result += hougharr[x][y][r];
			}
			temp.at<int>(y, x) = result;
		}
	}

	cv::normalize(temp, output, 0, 255, NORM_MINMAX);

	cv::imwrite("sobelout/hough.jpg", output);

	std::vector<tuple<int, int, int>> circles;
	for (int y = 0; y < dim2; y++) {
		for (int x = 0; x < dim1; x++) {
			if (temp.at<int>(y, x) > houghthresh) {
				for (int r = 0; r < dim3; r++) {
					int l = hougharr[x][y][r];
					if (l >= radthresh) {
						//std::cout << x << "," << y << "," << r << std::endl;
						tuple<int, int, int> circle = make_tuple(x + x0, y + y0, r + minr);
						circles.push_back(circle);
					}
				}
			}
		}
	}

	return circles;
}

std::pair<int, int> circleCenter(int radius, std::pair<int, int> pos, double grad) {
	int x = pos.first;
	int y = pos.second;
	int x0 = x + floor(radius * cos(grad));
	int y0 = y + floor(radius * sin(grad));


	return make_pair(x0, y0);
}

vector<pair<int, int>> houghTransformLines(Mat treshholded, Mat& dir, Mat& frame, int x0dim, int y0dim, int x1dim, int y1dim, int houghthresh, int lineintersectionthresh) {
	int thetaRange = 180;
	int rhoRange = 2 * diagonalOfImage(y1dim-y0dim+1, x1dim-x0dim+1);
	Mat houghArr2D(2 * diagonalOfImage(y1dim-y0dim+1, x1dim-x0dim+1), thetaRange, DataType<int>::type, Scalar(0));
	int theta;
	int rho;
	int wiggledDirection;
	int perpDirection;
	int thetaWiggle;

	//hough
	for (int y = 0; y < y1dim +1 -y0dim; y++) {
		for (int x = 0; x < x1dim + 1 - x0dim; x++) {
			if (treshholded.at<int>(y + y0dim, x + x0dim) == 255) {

				for (thetaWiggle = -1; thetaWiggle < 2; thetaWiggle++) {
					wiggledDirection = fmod((dir.at<double>(y + y0dim, x + x0dim) * 180 / M_PI + thetaWiggle + 180), 180);
					rho = (x) * cos(wiggledDirection * M_PI / 180.0) + (y) * sin(wiggledDirection * M_PI / 180.0);
					houghArr2D.at<int>(rho + diagonalOfImage(y1dim - y0dim + 1, x1dim - x0dim + 1), wiggledDirection) += 1;
				}
			}
		}
	}

	//threshold hough
	Mat normalizedHough(diagonalOfImage(y1dim - y0dim + 1, x1dim - x0dim + 1), thetaRange, DataType<int>::type, Scalar(0));
	cv::normalize(houghArr2D, normalizedHough, 0, 255, NORM_MINMAX);
	imwrite("houghout/hough.jpg", normalizedHough);

	Mat treshHoldedHough = tresholdHough2D(normalizedHough, rhoRange, thetaRange, houghthresh); //Hough Threshold
	imwrite("houghout/houghTreshed.jpg", treshHoldedHough);

	//get the lines from the hough
	std::vector<tuple<int, int>> lines;
	for (int i = 0; i < rhoRange; i++) {
		for (int j = 0; j < thetaRange; j++) {
			if (treshHoldedHough.at<int>(i, j) == 255) {
				tuple<int, int> line = make_tuple(i, j);
				lines.push_back(line);
			}
		}
	}

	//plot lines on output image
	float rhoToReconstruct;
	float thetaToReconstruct;
	float a, b, x0, y0, x1, y1, x2, y2;
	for (int i = 0; i < lines.size(); i++) {
		rhoToReconstruct = get<0>(lines.at(i)) - diagonalOfImage(y1dim - y0dim + 1, x1dim - x0dim + 1);
		thetaToReconstruct = get<1>(lines.at(i));

		a = cos(thetaToReconstruct * M_PI / 180.0);
		b = sin(thetaToReconstruct * M_PI / 180.0);

		x0 = a * rhoToReconstruct;
		y0 = b * rhoToReconstruct;

		x1 = int(x0 + 1000 * (-b));
		y1 = int(y0 + 1000 * (a));

		x2 = int(x0 - 1000 * (-b));
		y2 = int(y0 - 1000 * (a));

	}

	vector<pair<int, int>> lineIntersections = findLineIntersections(lines, treshholded, lineintersectionthresh, x0dim, y0dim, x1dim, y1dim);

	//plot line intersections
	//pair<int, int> toPlot;
	//for (int i = 0; i < lineIntersections.size(); i++) {
	//	toPlot = lineIntersections.at(i);
	//	circle(frame, Point(toPlot.first, toPlot.second), 1, (0, 0, 0), 10, CV_FILLED, 0);
	//}
	//imshow("lines", frame);
	//waitKey(0);
	return lineIntersections;
}

vector<pair<int, int>> findLineIntersections(std::vector<tuple<int, int>> lines, Mat treshholded, int lineintersectionthresh, int x0, int y0,int x1, int y1) {
	Mat matToFindIntersections(y1 - y0 + 1, x1 - x0 + 1, DataType<int>::type, Scalar(0));
	float rhoToReconstruct, thetaToReconstruct, thetaToReconstructRadians;
	int x, y;
	float m, b;

	//Plot the lines on empty Mat matching the image.
	for (int i = 0; i < lines.size(); i++) {
		rhoToReconstruct = get<0>(lines.at(i)) - diagonalOfImage(y1 - y0 + 1, x1 - x0 + 1);
		thetaToReconstruct = get<1>(lines.at(i));
		thetaToReconstructRadians = thetaToReconstruct * M_PI / 180.0;
		if (sin(thetaToReconstructRadians) != 0) {
			m = -(cos(thetaToReconstructRadians) / sin(thetaToReconstructRadians));
			for (x = 0; x < x1 - x0 + 1; x++) {
				y = -(cos(thetaToReconstructRadians) / sin(thetaToReconstructRadians)) * x + rhoToReconstruct / sin(thetaToReconstructRadians);
				if ((y >= 0) && (y < y1 - y0 + 1)) {
					matToFindIntersections.at<int>(y, x) += 1;
				}
			}
		}
	}

	//threshold the line intersections
	Mat normalizedIntersections(y1 - y0 + 1, x1- x0 + 1, DataType<int>::type, Scalar(0));
	cv::normalize(matToFindIntersections, normalizedIntersections, 0, 255, NORM_MINMAX);
	imwrite("houghout/normalizedIntersections.jpg", normalizedIntersections);
	imwrite("linesout/lines.jpg", normalizedIntersections);
	Mat threshedLineIntersections = thresholdLineIntersections(normalizedIntersections, lineintersectionthresh);	//threshing the line intersections
	imwrite("linesout/lineIntersections.jpg", threshedLineIntersections);

	//get the thresholded intersection points coordinates
	vector<pair<int, int>> threshedLineIntersectionPoints;
	for (int i = 0; i < threshedLineIntersections.rows; i++) {
		for (int j = 0; j < threshedLineIntersections.cols; j++) {
			if (threshedLineIntersections.at<int>(i, j) == 255) {
				threshedLineIntersectionPoints.push_back(make_pair(j + x0, i + y0));
			}
			else threshedLineIntersections.at<int>(i, j) = 0;
		}
	}

	//group the intersection points
	vector<pair<int, int>>groupedIntersectionCenters = intersectionCenters(threshedLineIntersectionPoints);
	printf("number of intersection points: %d\n", groupedIntersectionCenters.size());

	return groupedIntersectionCenters;
}

vector<pair<int, int>> intersectionCenters(vector<pair<int, int>> lines) {
	int centers = 0;
	vector<pair<int, int>> center_means;
	for (int i = 0; i < lines.size(); i++) {
		int x = get<0>(lines[i]);
		int y = get<1>(lines[i]);
		if (i == 0) {
			centers++;
			center_means.push_back(make_pair(x, y));
		}
		else {
			float min = 10000.0;
			pair<int, int> p = make_pair(0, 0);
			for (int j = 0; j < center_means.size(); j++) {
				int mx = center_means[j].first;
				int my = center_means[j].second;
				float dist = sqrt((mx - x) * (mx - x) + (my - y) * (my - y));
				if (dist < min) {
					min = dist;
					p = make_pair(mx, my);
				}
			}
			if (min > 10.0) {
				centers++;
				center_means.push_back(make_pair(x, y));
			}
			else {
				int index = 0;
				for (int j = 0; j < center_means.size(); j++) {
					if (center_means[j] == p) {
						index = j;
					}
				}
				center_means[index].first = floor((center_means[index].first + x) / 2);
				center_means[index].second = floor((center_means[index].second + y) / 2);
			}
		}
	}
	return center_means;
}

Mat thresholdLineIntersections(Mat normalizedLineIntersections, int threshold) {
	for (int i = 0; i < normalizedLineIntersections.rows; i++) {
		for (int j = 0; j < normalizedLineIntersections.cols; j++) {
			if (normalizedLineIntersections.at<int>(i, j) > threshold) {
				normalizedLineIntersections.at<int>(i, j) = 255;
			}
			else normalizedLineIntersections.at<int>(i, j) = 0;
		}
	}
	return normalizedLineIntersections;
}

tuple<Mat, Mat, Mat, Mat> sobelReturn(Mat& image) {
	Mat padded;
	cv::copyMakeBorder(image, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
	Mat dx = sobeldx(padded);
	Mat dy = sobeldy(padded);
	Mat grad = sobelmag(dx, dy);
	pair<Mat, Mat> p = sobeldir(dx, dy);
	Mat dir = p.first;
	return make_tuple(dx, dy, grad, dir);
}


Mat sobeldx(cv::Mat& image) {
	Mat temp(image.rows - 2, image.cols - 2, DataType<int>::type, Scalar(0));
	Mat output(image.rows - 2, image.cols - 2, DataType<int>::type, Scalar(0));
	for (int y = 1; y < temp.rows + 1; y++) {
		for (int x = 1; x < temp.cols + 1; x++) {
			int result = 0;
			result += -(image.at<uchar>(y - 1, x - 1));
			result += -2 * (image.at<uchar>(y, x - 1));
			result += -(image.at<uchar>(y + 1, x - 1));
			result += (image.at<uchar>(y - 1, x + 1));
			result += 2 * (image.at<uchar>(y, x + 1));
			result += (image.at<uchar>(y + 1, x + 1));
			temp.at<int>(y - 1, x - 1) = (int)result;
		}
	}
	cv::normalize(temp, output, 0, 255, NORM_MINMAX);
	return temp;
}

Mat sobeldy(cv::Mat& image) {
	Mat temp(image.rows - 2, image.cols - 2, DataType<int>::type, Scalar(0));
	Mat output(image.rows - 2, image.cols - 2, DataType<int>::type, Scalar(0));
	for (int y = 1; y < temp.rows + 1; y++) {
		for (int x = 1; x < temp.cols + 1; x++) {
			int result = 0;
			result += -(image.at<uchar>(y - 1, x - 1));
			result += -2 * (image.at<uchar>(y - 1, x));
			result += -(image.at<uchar>(y - 1, x + 1));
			result += (image.at<uchar>(y + 1, x + 1));
			result += 2 * (image.at<uchar>(y + 1, x));
			result += (image.at<uchar>(y + 1, x - 1));
			temp.at<int>(y - 1, x - 1) = (int)result;
		}
	}
	cv::normalize(temp, output, 0, 255, NORM_MINMAX);
	return temp;
}


cv::Mat sobelmag(cv::Mat& dx, cv::Mat& dy) {
	Mat temp(dx.rows, dx.cols, DataType<float>::type, Scalar(0));
	Mat output(dx.rows, dx.cols, DataType<float>::type, Scalar(0));
	for (int y = 0; y < dx.rows; y++) {
		for (int x = 0; x < dx.cols; x++) {
			int tbs = dy.at<int>(y, x) * dy.at<int>(y, x) + dx.at<int>(y, x) * dx.at<int>(y, x);
			temp.at<float>(y, x) = sqrt(tbs);
		}
	}
	cv::normalize(temp, output, 0, 255, NORM_MINMAX);
	return temp;
}

pair<Mat, Mat> sobeldir(cv::Mat& dx, cv::Mat& dy) {
	Mat temp(dx.rows, dx.cols, DataType<double>::type, Scalar(0));
	Mat output(dx.rows, dx.cols, DataType<double>::type, Scalar(0));
	for (int y = 0; y < dx.rows; y++) {
		for (int x = 0; x < dx.cols; x++) {
			temp.at<double>(y, x) = atan2(dy.at<int>(y, x), dx.at<int>(y, x));
		}
	}
	cv::normalize(temp, output, 0, 255, NORM_MINMAX);
	return make_pair(temp, output);
}

int*** malloc3dArray(int dim1, int dim2, int dim3)
{
	int i, j, k;
	int*** array = (int***)malloc(dim1 * sizeof(int**));

	for (i = 0; i < dim1; i++) {
		array[i] = (int**)malloc(dim2 * sizeof(int*));
		for (j = 0; j < dim2; j++) {
			array[i][j] = (int*)malloc(dim3 * sizeof(int));
		}

	}

	for (i = 0; i < dim1; ++i)
		for (j = 0; j < dim2; ++j)
			for (k = 0; k < dim3; ++k)
				array[i][j][k] = 0;

	return array;
}
/** @function detectAndDisplay */
std::vector<Rect> detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

	// 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

	 //4. Draw box around faces found
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 255), 2);
		std::cout <<"Viola-Jones:"<< faces[i].x << ", " << faces[i].y << ", " << faces[i].x + faces[i].width << "," << faces[i].y + faces[i].height << std::endl;
	}
	return faces;
}