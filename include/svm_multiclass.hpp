/*====================================================================================
*Wrapper for OpenCV SVM_Multiclass
*author: Carlos Caetano
*contact: carlos.caetano@dcc.ufmg.br
======================================================================================*/

#ifndef SVM_MULTICLASS_H
#define SVM_MULTICLASS_H

#include "classification.hpp"
#include <opencv2\opencv.hpp>
#include <xfunctional>
#include <fstream>
//#include <opencv2\ml.hpp>

// SVM model recompiled to get the decision function values (scores)
#include "ml_openCV_recomp.hpp"

using namespace cv::ml;

namespace ccr{

	class SVM_Multiclass : public Classification {
		int hardMiningMaxIt;
		bool hardNegMining_;
		void trainSVM(const cv::Mat &samples, const std::vector<std::string> &labelVec);
	protected:
		cv::Ptr<SVM> model_;
		bool trainAuto_;
		float weight1_;
		float weight2_;

	public:
		SVM_Multiclass(cv::FileStorage &storage);
		~SVM_Multiclass();

		void beforeProcess();

		// Classify samples and set responses.
		void predict(const cv::Mat &X, cv::Mat &responses);

		// learn the classifier
		void learn();

		// save classifier 
		void save(cv::FileStorage &storage);

		// load classifier
		void load(const cv::FileNode &node, cv::FileStorage &storage);

		virtual Classification* duplicateParameters();
	};

	//Functions from precomp.hpp and inner_functions.cpp
	void cvPreparePredictData(const CvArr* sample, int dims_all, const CvMat* comp_idx,
		int class_count, const CvMat* prob, float** row_sample,
		int as_sparse CV_DEFAULT(0));

	static int icvCmpSparseVecElems(const void* a, const void* b);

	inline void allocateSumMatrix(double **&matrix, int size);
	inline void deleteSumMatrix(double **&matrix, int size);

}



#endif
