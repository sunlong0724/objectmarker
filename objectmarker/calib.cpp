#include "stdafx.h"
#include "stl_import.h"
#include<opencv2/opencv.hpp>
#include <time.h>



CvMat * getProjectMatrix(CvMat * cam_intrinsic_matrix,CvMat * r,CvMat * t)
{
	CvMat * project_m = cvCreateMat(3,4,CV_32FC1);

	CvMat * intrinsic = cvCreateMat(3,4,CV_32FC1);
	CvMat * RT = cvCreateMat(4,4,CV_32FC1);
	CvMat * R = cvCreateMat(3,3,CV_64FC1);


	cvSetZero(project_m);
	cvSetZero(intrinsic);
	cvSetZero(RT);

	::cvRodrigues2(r,R);
	

	intrinsic->data.fl[0] = cam_intrinsic_matrix->data.fl[0];
	intrinsic->data.fl[2] = cam_intrinsic_matrix->data.fl[2];
	intrinsic->data.fl[5] = cam_intrinsic_matrix->data.fl[4];
	intrinsic->data.fl[6] = cam_intrinsic_matrix->data.fl[5];
	intrinsic->data.fl[10] = cam_intrinsic_matrix->data.fl[8];

	RT->data.fl[0] = R->data.db[0];
	RT->data.fl[1] = R->data.db[1];
	RT->data.fl[2] = R->data.db[2];
	RT->data.fl[3] = t->data.db[0];
	RT->data.fl[4] = R->data.db[3];
	RT->data.fl[5] = R->data.db[4];
	RT->data.fl[6] = R->data.db[5];
	RT->data.fl[7] = t->data.db[1];
	RT->data.fl[8] = R->data.db[6];
	RT->data.fl[9] = R->data.db[7];
	RT->data.fl[10] = R->data.db[8];
	RT->data.fl[11] = t->data.db[2];
	RT->data.fl[15] = 1;

	cvMatMul(intrinsic,RT,project_m);
	//cvSave("intrinsic.xml",intrinsic);
	//cvSave("R.xml",RT);
	//cvSave("P.xml",project_m);

	cvReleaseMat(&intrinsic);
	cvReleaseMat(&RT);
	cvReleaseMat(&R);

	return project_m;
}

vector<float> reproject3DPoint(CvMat * l_project, CvMat * r_project,float u1, float v1,float u2,float v2)
{
	vector<float> ret_point;
	CvMat * memm = cvCreateMat(4,3,CV_32FC1);
	CvMat * memm_r = cvCreateMat(3,1,CV_32FC1);
	CvMat * r_point = cvCreateMat(3,1,CV_32FC1);

	cvSetZero(memm_r);

	float * _d = l_project->data.fl;
	memm->data.fl[0] = u1*_d[8] - _d[0];
	memm->data.fl[1] = u1*_d[9] - _d[1];
	memm->data.fl[2] = u1*_d[10] - _d[2];

	memm->data.fl[3] = v1*_d[8] - _d[4];
	memm->data.fl[4] = v1*_d[9] - _d[5];
	memm->data.fl[5] = v1*_d[10] - _d[6];

	_d = r_project->data.fl;
	memm->data.fl[6] =   u2*_d[8] - _d[0];
	memm->data.fl[7] = 	 u2*_d[9] - _d[1];
	memm->data.fl[8] = 	 u2*_d[10] - _d[2];

	memm->data.fl[9] = 	 v2*_d[8] - _d[4];
	memm->data.fl[10] =	 v2*_d[9] - _d[5];
	memm->data.fl[11] =	 v2*_d[10] - _d[6];

	_d = memm_r->data.fl;
	_d[0] = l_project->data.fl[3] - u1*l_project->data.fl[11];
	_d[1] = l_project->data.fl[7] - v1*l_project->data.fl[11];

	_d[2] = r_project->data.fl[3] - u2*r_project->data.fl[11];
	_d[3] = r_project->data.fl[7] - v2*r_project->data.fl[11];

	cvSolve(memm,memm_r,r_point,CV_LU );

	//cvSave("3dpoint.xml",r_point);

	ret_point.push_back(r_point->data.fl[0]);
	ret_point.push_back(r_point->data.fl[1]);
	ret_point.push_back(r_point->data.fl[2]);

	return ret_point;
}

void get_r_t_matrix_by_points(CvMat* out_r_matrix, CvMat* out_t_matrix, CvMat * cam_intrinsic_matrix, CvMat * cam_distortion_coeffs, float * point_2d, double * point_3d, int nPoint) {
	CvMat * ipt = cvCreateMat(nPoint, 2, CV_32FC1);
	CvMat * wpt = cvCreateMat(nPoint, 3, CV_64FC1);

	memcpy(ipt->data.fl, point_2d, nPoint * sizeof(float) * 2);
	memcpy(wpt->data.fl, point_3d, nPoint * sizeof(double) * 3);

	::cvFindExtrinsicCameraParams2(wpt, ipt, cam_intrinsic_matrix, cam_distortion_coeffs, out_r_matrix, out_t_matrix);// get R  T

	cvReleaseMat(&ipt);
	cvReleaseMat(&wpt);
}

//校准图像上的2d坐标
void adjust_2d_point_uv(CvMat * cam_intrinsic_matrix, CvMat * cam_distortion_coeffs,int x, int y, float& u, float& v) {
	CvMat * src_2dpoint = cvCreateMat(1, 1, CV_64FC2);
	CvMat * dst_2dpoint = cvCreateMat(1, 1, CV_64FC2);
	src_2dpoint->data.db[0] = x;
	src_2dpoint->data.db[1] = y;

	::cvUndistortPoints(src_2dpoint, dst_2dpoint, cam_intrinsic_matrix, cam_distortion_coeffs);

	CvMat * src_uv = cvCreateMat(3, 1, CV_32FC1);
	CvMat * dst_uv = cvCreateMat(3, 1, CV_32FC1);
	src_uv->data.fl[0] = dst_2dpoint->data.db[0];
	src_uv->data.fl[1] = dst_2dpoint->data.db[1];
	src_uv->data.fl[2] = 1;

	cvMatMul(cam_intrinsic_matrix, src_uv, dst_uv);

	u = dst_uv->data.fl[0] / dst_uv->data.fl[2];
	v = dst_uv->data.fl[1] / dst_uv->data.fl[2];

	cvRelease((void**)&src_2dpoint);
	cvRelease((void**)&dst_2dpoint);
	cvRelease((void**)&src_uv);
	cvRelease((void**)&dst_uv);
}


void cam_test()
{
	float cam_intrinsic[9]  = 
	{
		1443.19825,	0,			900.76779,
		0,			1446.37588,	552.74644,
		0,			0,			1
	};

	float cam_distortion[5] = 
	//	{0.00000};
	{  -0.11519,   -0.07809,   -0.01185,   -0.00344,  0.00000};


	float cam_intrinsic2[9]  = 
	{
		1498.04048,	0,			934.73521,
		0,			1506.41293,	554.47105,
		0,			0,			1
	};

	float cam_distortion2[5] = 
	//	{0.00000};
	{  -0.12837,   0.17892,   -0.00282,   -0.00046,  0.00000};

	CvMat * cam_intrinsic_matrix = cvCreateMat(3,3,CV_32FC1);
	CvMat * cam_distortion_coeffs= cvCreateMat(1,5,CV_32FC1);

	memcpy(cam_intrinsic_matrix->data.fl,cam_intrinsic,sizeof(float)*9);
	memcpy(cam_distortion_coeffs->data.fl,cam_distortion,sizeof(float)*5);


	CvMat * cam_intrinsic_matrix2 = cvCreateMat(3,3,CV_32FC1);
	CvMat * cam_distortion_coeffs2= cvCreateMat(1,5,CV_32FC1);

	memcpy(cam_intrinsic_matrix2->data.fl,cam_intrinsic,sizeof(float)*9);
	memcpy(cam_distortion_coeffs2->data.fl,cam_distortion,sizeof(float)*5);

	CvMat * mat_Img = cvCreateMat(5,2,CV_32F);
	CvMat * mat_World = cvCreateMat(5,3,CV_64F);

	float point_2d_1[10] = {
		949,766,
		1568,555,
		1496,264,
		1203,161,
		453,266
	};

	double point_3d_1[15] = {
		2552, 4000,0,     	
		6600, 4000,0,    		
		10618,8000,0,    		
		10618,12000,0,    		
		2552,12000,0		
	};


	float point_2d_2[10] = {
		1418,369,
		675,218,	
		376,307,	
		273,603,	
		894,866		
	};
	double point_3d_2[15] = {
		2552,4000,0,
		10618,4000,0,
		10618,8000,0,
		6600,12000,0,
		2552,12000,0	
	};


	memcpy(mat_Img->data.fl,point_2d_1,sizeof(float)*10);
	memcpy(mat_World->data.fl,point_3d_1,sizeof(double)*15);

	//first point for r t matrix
	CvMat* r_matrix;
	CvMat* t_matrix;
	CvMat* mrt_matrix;
	r_matrix = cvCreateMat(3,1,CV_64F);
	t_matrix = cvCreateMat(3,1,CV_64F);
	get_r_t_matrix_by_points(r_matrix,t_matrix,cam_intrinsic_matrix,cam_distortion_coeffs, point_2d_1,point_3d_1,5);
	mrt_matrix = getProjectMatrix(cam_intrinsic_matrix, r_matrix, t_matrix);
	cvSave("r.xml", r_matrix);
	cvSave("t.xml", t_matrix);
	cvSave("mrt.xml", mrt_matrix);

	//second point for r t matrix
	CvMat* r_matrix_1;
	CvMat* t_matrix_1;
	CvMat* mrt_matrix_1;
	r_matrix_1 = cvCreateMat(3, 1, CV_64F);
	t_matrix_1 = cvCreateMat(3, 1, CV_64F);
	get_r_t_matrix_by_points(r_matrix_1, t_matrix_1, cam_intrinsic_matrix, cam_distortion_coeffs, point_2d_2, point_3d_2, 5);
	mrt_matrix_1 = getProjectMatrix(cam_intrinsic_matrix, r_matrix_1, t_matrix_1);
	cvSave("r1.xml", r_matrix_1);
	cvSave("t1.xml", t_matrix_1);
	cvSave("mrt1.xml", mrt_matrix_1);

	//3d重建
	//校准第一个相机图像上的2d坐标
	float left_u, left_v;
	adjust_2d_point_uv(cam_intrinsic_matrix, cam_distortion_coeffs, 1542, 519, left_u, left_v);

	float right_u, right_v;
	adjust_2d_point_uv(cam_intrinsic_matrix2, cam_distortion_coeffs2, 1542, 519, right_u, right_v);

	vector<float> vec;
	vec = reproject3DPoint(mrt_matrix, mrt_matrix_1, left_u, left_v, right_u, right_v);

	cvRelease((void**)&r_matrix);
	cvRelease((void**)&t_matrix);
	cvRelease((void**)&mrt_matrix);

	cvRelease((void**)&r_matrix_1);
	cvRelease((void**)&t_matrix_1);
	cvRelease((void**)&mrt_matrix_1);
}

int GAMMA_LUT[256];
void build_gamma_lut(float gamma)
{

	for (int i = 0; i < 256; i++)
	{
		float _s = i;
		if (_s == 0) _s == 1;

		_s /= 255;
		_s = pow(_s, gamma);
		GAMMA_LUT[i] = _s * 255;
	}
}

using namespace cv;

void gamma(cv::Mat img)
{


	//build_gamma_lut(0.8);

	for (int i = 0; i < img.rows; i++)
	{
		uchar * p_f = img.data + img.step * i;
		for (int j = 0; j < img.cols; j++)
		{

			p_f[0] = GAMMA_LUT[p_f[0]];
			p_f[1] = GAMMA_LUT[p_f[1]];
			p_f[2] = GAMMA_LUT[p_f[2]];

			p_f += img.channels();
		}
	}
}


//img_num:图像的组数
//corner_point_num:角点个数
//chessboard_w：棋盘格横轴方向的长度（棋盘格必须为正方形，所以横轴和纵轴长度相同）
int run_calibration(string& camera_id, int in_img_num, int in_corner_point_num_x, int in_chessboard_x, std::function<void(cv::Mat* mat)> showImage, std::function<bool(int,int)> check_corner) {

	build_gamma_lut(0.45f);

	CvMat*cam_object_points2;
	CvMat*cam_image_points2;
	int cam_board_n;
	int successes = 0;
	int img_num, cam_board_w, cam_board_h, cam_Dx, cam_Dy;


#if 1
	img_num = 10; //
	cam_board_w = 9;
	cam_board_h = 7;
	cam_board_n = cam_board_w * cam_board_h;
	cam_Dx = 105;
	cam_Dy = 105;
#endif

	cam_board_n = cam_board_w*cam_board_h;

	string str_pic_root_dir(get_pwd() + string("\\pic\\"));
	string xml_root_dir(get_pwd() + string("\\output\\xml"));
	string bmp_root_dir(get_pwd() + string("\\output\\bmp"));

	create_dir(str_pic_root_dir);
	create_dir(xml_root_dir);
	create_dir(bmp_root_dir);


	CvSize cam_board_sz = cvSize(cam_board_w, cam_board_h);
	CvMat*cam_image_points = cvCreateMat(cam_board_n*(img_num), 2, CV_32FC1);
	CvMat*cam_object_points = cvCreateMat(cam_board_n*(img_num), 3, CV_32FC1);
	CvMat*cam_point_counts = cvCreateMat((img_num), 1, CV_32SC1);
	CvPoint2D32f*cam_corners = new CvPoint2D32f[cam_board_n];
	int cam_corner_count;
	int cam_step;
	CvMat*cam_intrinsic_matrix = cvCreateMat(3, 3, CV_32FC1);
	CvMat*cam_distortion_coeffs = cvCreateMat(4, 1, CV_32FC1);
	CvSize cam_image_sz;
	//window intit
	//cvNamedWindow("window", 0);

	//get image size
	IplImage *cam_image_temp = cvLoadImage(string(str_pic_root_dir + "\\"+camera_id + "\\g1.jpg").c_str(), 0);
	cam_image_sz = cvGetSize(cam_image_temp);
	char failurebuf[20] = { 0 };
	/*
	//extract cornner
	// camera image
	//
	// pattern
	*/




	/*
	//extrat the cam cornner
	//
	//
	//
	*/
	fstream cam_data;
	cam_data.open("output\\TXT\\cam_corners.txt", ofstream::out);
	fstream cam_object_data;
	cam_object_data.open("output\\TXT\\cam_object_data.txt", ofstream::out);
	//process the prj image so that we can easy find cornner
	for (int ii = 1; ii < img_num; ii++)
	{
		char cambuf[1024] = { 0 };
		sprintf(cambuf, "%s\\%s\\g%d.jpg", str_pic_root_dir.c_str(), camera_id.c_str(), ii);
		cout << cambuf << endl;
		IplImage *cam_image_color = cvLoadImage(cambuf);
		IplImage * cam_image = cvCreateImage(cvGetSize(cam_image_color), 8, 1);
		cvCvtColor(cam_image_color, cam_image, CV_BGR2GRAY);

		//extract cam cornner
		int cam_found = cvFindChessboardCorners(cam_image, cam_board_sz, cam_corners, &cam_corner_count,
			CV_CALIB_CB_FILTER_QUADS);
		cout << "cvFindChessboardCorners" << cam_found << endl;
		cvFindCornerSubPix(cam_image, cam_corners, cam_corner_count,
			cvSize(11, 11), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cvDrawChessboardCorners(cam_image_color, cam_board_sz, cam_corners, cam_corner_count, cam_found);

		if (cam_corner_count != cam_board_n)
			cout << "find cam" << ii << "  corner failed!\n";

		showImage(&Mat(cam_image_color));
		if (!check_corner(cam_board_n, cam_corner_count)) {
			continue;
		}

		//when cam and prj are success store the result
		if (cam_corner_count == cam_board_n) {
			//store cam result
			cam_step = successes*cam_board_n;
			for (int i = cam_step, j = 0; j < cam_board_n; ++i, ++j) {
				CV_MAT_ELEM(*cam_image_points, float, i, 0) = cam_corners[j].x;
				CV_MAT_ELEM(*cam_image_points, float, i, 1) = cam_corners[j].y;
				CV_MAT_ELEM(*cam_object_points, float, i, 0) = (j / cam_board_w)*cam_Dx;
				CV_MAT_ELEM(*cam_object_points, float, i, 1) = (j % cam_board_w)*cam_Dy;
				CV_MAT_ELEM(*cam_object_points, float, i, 2) = 0.0f;
				cam_data << cam_corners[j].x << "\t" << cam_corners[j].y << "\n";
				cam_object_data << (j / cam_board_w)*cam_Dx << "\t" << (j %cam_board_w)*cam_Dy << "\t0\n";
			}
			CV_MAT_ELEM(*cam_point_counts, int, successes, 0) = cam_board_n;
			successes++;
			cout << "success number" << successes << endl;
			cvSaveImage((bmp_root_dir + "\\" + camera_id + "_" + std::to_string(ii) + "_chessboard.bmp").c_str(), cam_image_color);
			//cvSaveImage("chessboard.bmp", cam_image_color);
			//cvWaitKey(500);
		}

	}

	if (successes < 2)
		exit(0);
	/*
	//restore the success point
	*/
	//cam
	cam_image_points2 = cvCreateMat(cam_board_n*(successes), 2, CV_32FC1);
	cam_object_points2 = cvCreateMat(cam_board_n*(successes), 3, CV_32FC1);
	CvMat*cam_point_counts2 = cvCreateMat((successes), 1, CV_32SC1);
	for (int i = 0; i < successes*cam_board_n; ++i) {
		CV_MAT_ELEM(*cam_image_points2, float, i, 0) = CV_MAT_ELEM(*cam_image_points, float, i, 0);
		CV_MAT_ELEM(*cam_image_points2, float, i, 1) = CV_MAT_ELEM(*cam_image_points, float, i, 1);
		CV_MAT_ELEM(*cam_object_points2, float, i, 0) = CV_MAT_ELEM(*cam_object_points, float, i, 0);
		CV_MAT_ELEM(*cam_object_points2, float, i, 1) = CV_MAT_ELEM(*cam_object_points, float, i, 1);
		CV_MAT_ELEM(*cam_object_points2, float, i, 2) = CV_MAT_ELEM(*cam_object_points, float, i, 2);

	}
	for (int i = 0; i < successes; ++i) {
		CV_MAT_ELEM(*cam_point_counts2, int, i, 0) = CV_MAT_ELEM(*cam_point_counts, int, i, 0);
	}
	//cvSave("output\\XML\\cam_corners.xml", cam_image_points2);
	cvSave((xml_root_dir + "\\" + camera_id + "_corners.xml").c_str(), cam_image_points2);

	cvReleaseMat(&cam_object_points);
	cvReleaseMat(&cam_image_points);
	cvReleaseMat(&cam_point_counts);


	/*
	//calibration for camera
	//
	*/
	//calib for cam
	CV_MAT_ELEM(*cam_intrinsic_matrix, float, 0, 0) = 1.0f;
	CV_MAT_ELEM(*cam_intrinsic_matrix, float, 1, 1) = 1.0f;
	CvMat* cam_rotation_all = cvCreateMat(successes, 3, CV_32FC1);
	CvMat* cam_translation_vector_all = cvCreateMat(successes, 3, CV_32FC1);
	cvCalibrateCamera2(
		cam_object_points2,
		cam_image_points2,
		cam_point_counts2,
		cam_image_sz,
		cam_intrinsic_matrix,
		cam_distortion_coeffs,
		cam_rotation_all,
		cam_translation_vector_all,
		0//CV_CALIB_FIX_ASPECT_RATIO  
		);

	CvMat * w_pts = cvCreateMat(48, 3, CV_32FC1);
	CvMat * i_pts = cvCreateMat(48, 2, CV_32FC1);

	memcpy(w_pts->data.fl, cam_object_points2->data.fl, sizeof(float) * 3 * 48);
	int rows = cam_rotation_all->rows;
	int rows2 = cam_translation_vector_all->rows;

	cam_rotation_all->rows = 1;
	cam_translation_vector_all->rows = 1;
	::cvProjectPoints2(w_pts, cam_rotation_all, cam_translation_vector_all, cam_intrinsic_matrix, cam_distortion_coeffs, i_pts);
	//cvSave("output\\XML\\reproject.xml", i_pts);
	cvSave((xml_root_dir + "\\" + camera_id + "_reproject.xml").c_str(), i_pts);

	cam_rotation_all->rows = rows;
	cam_translation_vector_all->rows = rows2;

	IplImage * pret = cvLoadImage((str_pic_root_dir + "\\" + camera_id + "\\g1.jpg").c_str());
	//IplImage * pret = cvLoadImage("cam\\g1.jpg");
	IplImage * pdst = cvCloneImage(pret);
	::cvUndistort2(pret, pdst, cam_intrinsic_matrix, cam_distortion_coeffs);

	//cvSaveImage("undistort.bmp", pdst);
	//cvSave("output\\XML\\cam_intrinsic_matrix.xml", cam_intrinsic_matrix);
	//cvSave("output\\XML\\cam_distortion_coeffs.xml", cam_distortion_coeffs);

	cvSaveImage((bmp_root_dir + "\\" + camera_id + "_undistort.bmp").c_str(), pdst);
	cvSave((xml_root_dir + "\\" + camera_id + "_intrinsic_matrix.xml").c_str(), cam_intrinsic_matrix);
	cvSave((xml_root_dir + "\\" + camera_id + "_distortion_coeffs.xml").c_str(), cam_distortion_coeffs);

	//calib 
	//cvSave("output\\XML\\cam_rotation_all.xml", cam_rotation_all);
	//cvSave("output\\XML\\cam_translation_vector_all.xml", cam_translation_vector_all);
	//char path1[100] = "output\\result_data_no_optim.txt";

	cvSave((xml_root_dir + "\\" + camera_id + "cam_rotation_all.xml").c_str(), cam_rotation_all);
	cvSave((xml_root_dir + "\\" + camera_id + "cam_translation_vector_all.xml").c_str(), cam_translation_vector_all);

	return 0;
}

