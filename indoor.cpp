#include <iostream>
#include <fstream>
#include <set>
#include <vector>
#include <iterator>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

// Mat(rows, cols), cols major order. (col by col append), new format
cv::Mat_<double> id_load(char * file)
{
    std::ifstream in(file);
    int i, j, cols, rows;
    double dummy;
    in >> cols >> rows >> dummy;
    Mat_<double> mat(rows,cols);
    double d;
	for (j=0; j<rows; ++j)
    for (i=0; i<cols; ++i) {
        in >> d;
        mat.at<double>(Point(i,j)) = d;
	}
    // return mat.t();
    return mat;
}

template <typename F>
void each(int n, F && f)
{
    for (int i=0; i<n; ++i)
        f(i);
}

template <typename F>
void each(int h, int w, F && f)
{
    for (int j=0; j<w; ++j)
    for (int i=0; i<h; ++i)
        f(i,j);
}

template <typename T, typename F>
void each(Mat_<T> & mat, int x0, int x1, int y0, int y1, F && f)
{
    x0 = x0<0 ? 0 : x0;
    y0 = y0<0 ? 0 : y0;
    x1 = x1>mat.rows ? mat.rows : x1;
    y1 = y1>mat.cols ? mat.cols : y1;
    for (int y=y0; y<y1; ++y)
    for (int x=x0; x<x1; ++x)
        f(x,y);
}


struct c_line { 
	Vec2f p[2]; 
	float len() { return (float)cv::norm(p[1] - p[0]); }
	Vec2f mid() { return (p[0] + p[1]) * 0.5; }
	Vec2f dir() { return (p[1] - p[0]) * (1 / len()); }
};

Vec2f xdir(Vec2f dir) { return Vec2f(dir[1], -dir[0]); }

c_line to_line(const Vec4i &l) {
	c_line line;
	Vec2i a(&l[0]), b(&l[2]);
	line.p[0] = Vec2f(a);
	line.p[1] = Vec2f(b);
	return line;
}

struct c_seg {
	float x, y;
	c_seg(float a, float b) { x = min(a,b); y = max(a,b); }
} ;

bool merge(c_seg a, c_seg b, float gap, c_seg &seg) {
	if (a.x > b.x)  std::swap(a,b); // make sure a.x <= b.x
	if (b.x - a.y < gap) 
		seg = c_seg(a.x, max(a.y, b.y));    // overlap
	else
		return false;
	return true;
}

class c_lineprocess {
public:
    c_lineprocess(vector<Vec4i> &lines) {
        _lines.resize(lines.size());
        for (unsigned int i=0; i<lines.size(); ++i)
            _lines[i] = to_line(lines[i]);
	}

    // all lines must align with primary directions
    int p10_rectify(std::vector<Vec2f> & pdirs, float thd) {
        std::vector<c_line> lines;
        lines.reserve(_lines.size());
        _pdirs = pdirs;
        _dirs.clear();
        _dirs.reserve(_lines.size());
        for (unsigned int i=0; i<_lines.size(); ++i) {
            c_line & l = _lines[i];
            c_line rsl;
            for (unsigned int j=0; j<pdirs.size(); ++j)
                if (fabs(pdirs[j].dot(l.dir())) > thd) { // bingo, find the closed primary direction
                    Vec2f m = l.mid();
                    float len = fabs((l.p[1] - l.p[0]).dot(pdirs[j]));
                    rsl.p[0] = m - len * 0.5 * pdirs[j];
                    rsl.p[1] = m + len * 0.5 * pdirs[j];
                    lines.push_back(rsl);
                    _dirs.push_back(j);
                    break;
				}
		}
        std::swap(_lines, lines);
        assert(_lines.size() == _dirs.size());
        std::cout << "p10_rectify, return " << _lines.size() << "\n";
        return _lines.size();
	}

    // merge line with same direction
    int p20_snapping(float thd1, float thd_gap) {
        std::vector<int> merged(_lines.size(), -1);
        for (unsigned int i=0; i<_lines.size()-1; ++i)
            for (unsigned int j=i+1; j<_lines.size(); ++j) {
                c_line & a = _lines[i];
                c_line & b = _lines[j];
                if (_dirs[i] != _dirs[j])
                    continue;
                Vec2f dir = _pdirs[_dirs[i]];
                Vec2f xdir(dir[1], -dir[0]);    // 90 degree vector
                float rhoa = xdir.dot(a.p[0]);
                float rhob = xdir.dot(b.p[0]);
                if (fabs(rhoa - rhob) > thd1)
                    continue;   //  distance too much 
                c_seg sega(a.p[0].dot(dir), a.p[1].dot(dir));
                c_seg segb(b.p[0].dot(dir), b.p[1].dot(dir));
                c_seg seg(0,0);
                if (!merge(sega, segb, thd_gap, seg))
                    continue;
                // merge i j to j (for further merging)
                float lena = a.len(), lenb = b.len();
                float rho = (lena * rhoa + lenb * rhob) / (lena + lenb);
                b.p[0] = xdir * rho + seg.x * dir;
                b.p[1] = xdir * rho + seg.y * dir;
                merged[i] = j;
			}
        std::vector<c_line> lines;
        std::vector<int> dirs;
        lines.reserve(_lines.size());
        dirs.reserve(_lines.size());
        for (unsigned int i=0; i<_lines.size(); ++i)
            if (merged[i] < 0) {
                lines.push_back(_lines[i]);
                dirs.push_back(_dirs[i]);
			}
        std::swap(_lines, lines);
        std::swap(_dirs, dirs);

        std::cout << "p20_snapping, return " << _lines.size() << "\n";
        return _lines.size();
	}

    // connect end points or nearsst line if distance < r
    // cross threshold 0.1, parallel threshold 0.95
    int p30_connect(float r) {
        std::cout << "p30_connect " << r << "\n";

        std::vector<c_line> lines(_lines);
        lines.reserve(lines.size() * 2);

        // the perpendicular direction of dir
        std::vector<int> xdirs(2);  // only 2 primary direction
        xdirs[0] = 1; xdirs[1] = 0;

        std::cout << "first, connect all cross lines\n";
        for (unsigned int i=0; i<_lines.size(); ++i) {
            c_line & line_i = _lines[i];
			Vec2f dir_i = line_i.dir();
            float ext[2];
            for (unsigned int j=0; j<_lines.size(); ++j) {
                if (i==j)
                    continue;
                c_line & line_j = _lines[j];
                Vec2f dir_j = line_j.dir();
                if (dir_i.dot(dir_j) > 0.1) // not perpendicular/cross
                    continue;

				ext[0] = cross_conn(line_i.p[0], -dir_i, r, line_j);
				ext[1] = cross_conn(line_i.p[1], dir_i, r, line_j);
			}
            if (ext[0] > 0)
                line_i.p[0] -= ext[0] * dir_i;
            if (ext[1] > 0)
                line_i.p[1] += ext[0] * dir_i;
		}
	}

    float cross_conn(Vec2f p, Vec2f dir, float r, c_line l) {
        return 0;
	}

    std::vector<c_line> _lines;
    std::vector<Vec2f>  _pdirs;
    std::vector<int>    _dirs;
};

int id_main( int argc, char** argv )
{
    // Mat_<double> mat = id_load("e:\\tmp\\001\\newData3\\floorDepth.txt");
    Mat_<double> mat = id_load("C:/work/tmp/003.rsh/newData/floorDepth.txt");
    int w = mat.cols, h = mat.rows;
    std::cout << "mat.cols " << w << ", mat.rows " << h << ", width " << mat.size().width << ", height " << mat.size().height << std::endl;
	imshow("source", mat);

    double threshold = 0, max_value = 0;
	{
		std::vector<double> vec;
		vec.reserve(w*h);
		each(h, w, [&](int i, int j) { vec.push_back(mat(i,j)); });
		std::sort(vec.begin(), vec.end());
		threshold = vec[(int)(vec.size() * 0.9)];
        max_value = *vec.rbegin();

        std::cout << "ordered image value " << vec.size() << " : \n";
        std::copy(vec.begin(), vec.begin()+10, std::ostream_iterator<double>(std::cout, ", "));
        std::cout << " ... ";
        std::copy(vec.end()-10, vec.end(), std::ostream_iterator<double>(std::cout, ", "));
        std::cout << std::endl;
		std::cout << "set threshold : " << threshold << std::endl;
		std::cout << "max value : " << max_value << std::endl;
	}

    std::cout << "build edge map for hough detection\n";
    Mat_<unsigned char> edge(h,w);
	each(h, w, [&](int i, int j) { edge(i,j) = mat(i,j) > threshold ? (unsigned char)(255 * mat(i,j)/max_value) : 0; });
	imshow("edge", edge);

    std::cout << "hough line segment search\n";
	vector<Vec4i> lines;
	HoughLinesP(edge, lines, 1, CV_PI/180, 5, 15, 5 );

    std::cout << "extract and draw line segments\n";
    Mat rsl(h, w, CV_8UC3);
    vector<Vec3f> lines_info(lines.size()); // cos/sin/len
    int num_of_dirs = 0;
	for( size_t i = 0; i < lines.size(); i++ )
	{
		cv::Vec4i l = lines[i];
        cv::Vec3f & info = lines_info[i];
        float dx = (float)(l[2] - l[0]), dy = (float)(l[3] - l[1]);
        float d = sqrt(dx*dx + dy * dy);
        info[0] = dx/d;  // cos(theta)
        info[1] = dy/d;  // sin(theta)
        info[2] = d;     // line seg length
        num_of_dirs += (int)ceil(d);

        // std::cout << "line " << i << " : " << lines[i] << std::endl;
        if (d > 30)
		line( rsl, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);

	}
	// each(lines.size(), [&](int i) { 
	// 	std::cout << "line seg " << i << " cos/sin " << lines_info[i] << ", pts " << lines[i] << std::endl; 
	// });
	imshow("result", rsl);

    std::cout << "draw direction circle map\n";
    Mat rsl_dir(800, 800, CV_8UC3);
    each(lines_info.size(), [&](int i) {
        cv::Vec3f &info = lines_info[i];
        int x = (int)(400 + info[2] * info[0]);
        int y = (int)(400 + info[2] * info[1]);
        line(rsl_dir, Point(400,400), Point(x,y), Scalar(0,0,255), 1, CV_AA);
	});

    std::vector<float> vdirs;
	vdirs.reserve(2*num_of_dirs);
    each(lines_info.size(), [&](int i) {
        cv::Vec3f &info = lines_info[i];
        int len = (int)ceil(info[2]);
        if (len < 30) return;
        auto & vd = vdirs;
        each(len, [&](int j) {   // points number is same as length
            vd.push_back(info[0]);
            vd.push_back(info[1]);
		});
	});
    Mat dirs(vdirs.size()/2, 2, CV_32F, &vdirs[0]);
    // address is same
    std::cout << "mat's element address : " << &dirs.at<float>(0,0) << ", vector address : " << &vdirs[0] << "\n";
    Mat labels, centers;
    std::vector<Vec2f> pdirs(2); // primary direction, 2 way
    kmeans(dirs, 2, labels, 
		cv::TermCriteria(CV_TERMCRIT_EPS, 100, 0), 
		10, KMEANS_RANDOM_CENTERS, centers);
    std::cout << "kmean result, labels size " << labels.size() << ", centers : \n" << centers << "\n";
    each(2, [&](int i){
        Vec2f & dir = centers.at<Vec2f>(i, 0);
        dir = dir * (1.0/(dir[0] * dir[0] + dir[1] * dir[1]));
        int x = (int)(400 + 200 * dir[0]);
        int y = (int)(400 + 200 * dir[1]);
		line(rsl_dir, Point(400,400), Point(x,y), Scalar(255,0,0), 1, CV_AA);
        pdirs[i] = dir;
	});
    std::cout << "two direction dot product " << centers.at<Vec2f>(0,0).dot(centers.at<Vec2f>(1,0)) << "\n";
	imshow("rsl_dir", rsl_dir);

    std::cout << "rectify all line segment.\n";
    c_lineprocess lp(lines);
    lp.p10_rectify(pdirs, 0.95f);
    lp.p20_snapping(10, 5);
    Mat rf_rsl(h, w, CV_8UC3);
    for (unsigned int i=0; i<lp._lines.size(); ++i) {
        c_line &l = lp._lines[i];
		line(rf_rsl, Point(l.p[0]), Point(l.p[1]), Scalar(255,0,0), 1, CV_AA);
	}
	imshow("50 - rectify", rf_rsl);

    std::cout << "detect corner\n";
    Mat corner = Mat::zeros( edge.size(), CV_32FC1);
	cornerHarris( edge, corner, 7, 5, 0.05, BORDER_DEFAULT );
    // Normalizing
    Mat corner_norm, corner_norm_scaled;
    normalize( corner, corner_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( corner_norm, corner_norm_scaled );
    std::cout << "get top 1% from corner\n";
    Mat rsl_corner(corner.size(), CV_8UC1);
    std::cout << "corner_norm_scaled, element type : " << corner_norm_scaled.type() << "\n";
	if (1) {
        std::vector<double> vec;
        vec.reserve(w*h);
		each(h, w, [&](int i, int j) { vec.push_back(corner_norm_scaled.at<unsigned char>(i,j)); });
        std::sort(vec.begin(), vec.end());
        double thd = vec[(int)(0.99 * vec.size())];
        double maxv = *vec.rbegin();
		each(h, w, [&](int i, int j) { 
			double c = corner_norm_scaled.at<unsigned char>(i,j);
			rsl_corner.at<unsigned char>(i,j) = c > thd ? (unsigned char)(255 * (c-thd)/(maxv-thd)) : 0;
		});
	}
	imshow("corner_scaled", corner_norm_scaled);
	imshow("rsl_corner", rsl_corner);

    cv::namedWindow( "source", cv::WINDOW_AUTOSIZE ); // Create a window for display.
    cv::waitKey(0); // Wait for a keystroke in the window

    return 0;
}
