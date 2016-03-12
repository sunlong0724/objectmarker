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
void adjust_2d_point_uv(CvMat * cam_intrinsic_matrix, CvMat * cam_distortion_coeffs,double x, double y, float& u, float& v) {
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


int calc_3d_point(const string& left_camera_ip,double left_u, double left_v, const string& right_camera_ip, float right_u, float right_v) {
	//3d重建
	string xml_root_dir(get_pwd() + string("\\output\\xml"));

	CvMat* left_cam_intrinsic_matrix;// = cvCreateMat(3, 3, CV_32FC1);
	CvMat* left_cam_distortion_coeffs;// = cvCreateMat(1, 5, CV_32FC1);
	CvMat* left_cam_r_matrix;
	CvMat* left_cam_t_matrix;
	CvMat* left_cam_mrt_matrix;

	CvMat* right_cam_intrinsic_matrix;// = cvCreateMat(3, 3, CV_32FC1);
	CvMat* right_cam_distortion_coeffs;// = cvCreateMat(1, 5, CV_32FC1);
	CvMat* right_cam_r_matrix;
	CvMat* right_cam_t_matrix;
	CvMat* right_cam_mrt_matrix;

	left_cam_intrinsic_matrix	= (CvMat*)cvLoad(string(xml_root_dir + "\\" + left_camera_ip + "_intrinsic_matrix.xml").c_str());
	left_cam_distortion_coeffs	= (CvMat*)cvLoad(string(xml_root_dir + "\\" + left_camera_ip + "_distortion_coeffs.xml").c_str());
	left_cam_r_matrix			= (CvMat*)cvLoad(string(xml_root_dir + "\\" + left_camera_ip + "_r.xml").c_str());
	left_cam_t_matrix			= (CvMat*)cvLoad(string(xml_root_dir + "\\" + left_camera_ip + "_t.xml").c_str());
	left_cam_mrt_matrix			= (CvMat*)cvLoad(string(xml_root_dir + "\\" + left_camera_ip + "_mrt.xml").c_str());

	right_cam_intrinsic_matrix	= (CvMat*)cvLoad(string(xml_root_dir + "\\" + right_camera_ip + "_intrinsic_matrix.xml").c_str());
	right_cam_distortion_coeffs = (CvMat*)cvLoad(string(xml_root_dir + "\\" + right_camera_ip + "_distortion_coeffs.xml").c_str());
	right_cam_r_matrix			= (CvMat*)cvLoad(string(xml_root_dir + "\\" + right_camera_ip + "_r.xml").c_str());
	right_cam_t_matrix			= (CvMat*)cvLoad(string(xml_root_dir + "\\" + right_camera_ip + "_t.xml").c_str());
	right_cam_mrt_matrix		= (CvMat*)cvLoad(string(xml_root_dir + "\\" + right_camera_ip + "_mrt.xml").c_str());

	float left_u_correct, left_v_correct;
	adjust_2d_point_uv(left_cam_intrinsic_matrix, left_cam_distortion_coeffs, left_u, left_v, left_u_correct, left_v_correct);

	float right_u_correct, right_v_correct;
	adjust_2d_point_uv(right_cam_intrinsic_matrix, right_cam_distortion_coeffs, right_u, right_v, right_u_correct, right_v_correct);

	//3d coordinate
	vector<float> vec;
	vec = reproject3DPoint(left_cam_mrt_matrix, right_cam_mrt_matrix, left_u, left_v, right_u, right_v);

	string data_root_dir(get_pwd() + string("\\output\\data"));
	create_dir(data_root_dir);
	string name(data_root_dir + "\\" + left_camera_ip + "_" + right_camera_ip + "_3d.txt");
	std::ofstream ofs(name, std::ofstream::out);
	if (ofs.fail()) {
		return -1;
	}
	char buff[1024];
	snprintf(buff, sizeof buff, "left(%f,%f), right(%f,%f), 3d(%f,%f,%f)\n", left_u, left_v, right_u, right_v, vec[0], vec[1], vec[2]);
	ofs.write(buff, strlen(buff));
	ofs.close();

	return 0;
}

int parse_point_from_file(const string& name, float* point_2d, double* point_3d, int size, int& count) {
	ifstream ifs(name, std::fstream::in);
	if (ifs.fail()) {
		return -1;
	}
	char cstr_line[1024];
	count = 0;
	int i = 0;
	int j = 0;
	while (ifs.getline(cstr_line, sizeof cstr_line, '\n')) {
		char* pch = strtok(cstr_line, ",");
		while (pch != NULL) {//five points per line
			point_2d[i++] = std::atof(pch);
			pch = strtok(NULL, ",");
			point_2d[i++] = std::atof(pch);
			pch = strtok(NULL, ",");
			point_3d[j++] = std::atof(pch);
			pch = strtok(NULL, ",");
			point_3d[j++] = std::atof(pch);
			pch = strtok(NULL, ",");
			point_3d[j++] = std::atof(pch);
			pch = strtok(NULL, ",");
		}
		++count;
	}
	ifs.close();
	return count;
}

int calc_extrinsic_camera_params(const string& camera_ip ) {

	const static int MAX_POINT_COUNT = 20;//20 points at most!!

	float point_2d_1[MAX_POINT_COUNT * 2] = {};
	double point_3d_1[MAX_POINT_COUNT * 3] = {};

	CvMat * cam_intrinsic_matrix;// = cvCreateMat(3, 3, CV_32FC1);
	CvMat * cam_distortion_coeffs;// = cvCreateMat(1, 5, CV_32FC1);

	CvMat* r_matrix;
	CvMat* t_matrix;
	CvMat* mrt_matrix;

	string xml_root_dir(get_pwd() + string("\\output\\xml"));
	string data_root_dir(get_pwd() + string("\\output\\data"));

	create_dir(xml_root_dir);

	cam_intrinsic_matrix = (CvMat*)cvLoad(string(xml_root_dir + "\\" + camera_ip + "_intrinsic_matrix.xml").c_str());
	cam_distortion_coeffs = (CvMat*)cvLoad(string(xml_root_dir + "\\" + camera_ip + "_distortion_coeffs.xml").c_str());

	int count = 0;
	if (0 > parse_point_from_file(data_root_dir + "\\" + camera_ip + ".jpg.txt", point_2d_1, point_3d_1, sizeof(point_2d_1) / sizeof(float), count))
		return -1;

	r_matrix = cvCreateMat(3, 1, CV_64F);
	t_matrix = cvCreateMat(3, 1, CV_64F);
	get_r_t_matrix_by_points(r_matrix, t_matrix, cam_intrinsic_matrix, cam_distortion_coeffs, point_2d_1, point_3d_1, count);
	mrt_matrix = getProjectMatrix(cam_intrinsic_matrix, r_matrix, t_matrix);

	string name_prefix{ xml_root_dir + "\\" + string(camera_ip) };
	cvSave( (name_prefix + "_r.xml").c_str(), r_matrix);
	cvSave((name_prefix + "_t.xml").c_str(), t_matrix);
	cvSave((name_prefix + "_mrt.xml").c_str(), mrt_matrix);

	return 0;
}


//img_num:图像的组数
//corner_point_num:角点个数
//chessboard_w：棋盘格横轴方向的长度（棋盘格必须为正方形，所以横轴和纵轴长度相同）
int run_calibration(string& camera_id, int in_img_num, int in_corner_point_num_x, int in_chessboard_x, std::function<void(cv::Mat* mat)> showImage, 
	std::function<bool(int,int)> check_corner,
	std::function<void(const string&)> display_caption
	) {

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
	//process the prj image so that we can easy find cornner
	for (int ii = 1; ii < img_num; ii++)
	{
		char cambuf[1024] = { 0 };
		sprintf(cambuf, "%s\\%s\\g%d.jpg", str_pic_root_dir.c_str(), camera_id.c_str(), ii);

		display_caption(cambuf);

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
			}
			CV_MAT_ELEM(*cam_point_counts, int, successes, 0) = cam_board_n;
			successes++;
			cout << "success number" << successes << endl;
			cvSaveImage((bmp_root_dir + "\\" + camera_id + "_" + std::to_string(ii) + "_chessboard.bmp").c_str(), cam_image_color);
			//cvSaveImage("chessboard.bmp", cam_image_color);
			//cvWaitKey(500);
		}

	}

	if (successes < 2){
		return -2;
	}
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

vector<float> project_3d_to_2d(const string& camera_ip, double x, double y, double z) {

	//::cvProjectPoints2(wpt, r_matrix, t_matrix, cam_intrinsic_matrix, cam_distortion_coeffs, ipt);

	string xml_root_dir(get_pwd() + string("\\output\\xml"));

	CvMat* cam_intrinsic_matrix;// = cvCreateMat(3, 3, CV_32FC1);
	CvMat* cam_distortion_coeffs;// = cvCreateMat(1, 5, CV_32FC1);
	CvMat* cam_r_matrix;
	CvMat* cam_t_matrix;
	CvMat* cam_mrt_matrix;

	CvMat * ipt= cvCreateMat(1, 2, CV_32FC1);
	CvMat * wpt = cvCreateMat(1, 3, CV_64FC1);
	wpt->data.db[0] = x;
	wpt->data.db[1] = y;
	wpt->data.db[2] = z;

	cam_intrinsic_matrix = (CvMat*)cvLoad(string(xml_root_dir + "\\" + camera_ip + "_intrinsic_matrix.xml").c_str());
	cam_distortion_coeffs = (CvMat*)cvLoad(string(xml_root_dir + "\\" + camera_ip + "_distortion_coeffs.xml").c_str());
	cam_r_matrix = (CvMat*)cvLoad(string(xml_root_dir + "\\" + camera_ip + "_r.xml").c_str());
	cam_t_matrix = (CvMat*)cvLoad(string(xml_root_dir + "\\" + camera_ip + "_t.xml").c_str());
	cam_mrt_matrix = (CvMat*)cvLoad(string(xml_root_dir + "\\" + camera_ip + "_mrt.xml").c_str());

	::cvProjectPoints2(wpt, cam_r_matrix, cam_t_matrix, cam_intrinsic_matrix, cam_distortion_coeffs, ipt);

	vector<float> ret = {ipt->data.fl[0], ipt->data.fl[1]};

	cvRelease((void**)&ipt);
	cvRelease((void**)&wpt);

	return ret;
}

void get_homography(float x1, float y1, float x2, float y2) {

	CvMat * ipt = cvCreateMat(1, 2, CV_32FC1);
	CvMat * wpt = cvCreateMat(1, 2, CV_32FC1);
	ipt->data.fl[0] = x1;
	ipt->data.fl[1] = y1;

	wpt->data.fl[0] = x1;
	wpt->data.fl[1] = y1;

	CvMat* homography;
	cvFindHomography(ipt, wpt, homography, CV_RANSAC);
	cvSave("H.xml", homography);

	cvRelease((void**)(&ipt));
	cvRelease((void**)(&wpt));
}