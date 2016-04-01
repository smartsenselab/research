#include "svm_multiclass.hpp"

///the first sample's label is considered the positive label

namespace ccr
{

	SVM_Multiclass::SVM_Multiclass(cv::FileStorage &storage) : Classification(storage){
		trainAuto_ = true; //false
		hardNegMining_ = false; //true
		hardMiningMaxIt = 12;
		
		model_ = SVM::create();
		beforeProcess();
	}


	SVM_Multiclass::~SVM_Multiclass(void){
		
	}


	void SVM_Multiclass::beforeProcess(){

		cv::FileNode node;

		std::string svm_type;
		std::string kernelType;
		int epsilon = 0;
		float C_;
		float nu_;
		float gamma_;
		float coef_;
		float degree_;
		float p_;
		int iteration_count;

		node = params["svm"];

		node["svm_type"] >> svm_type;
		node["C"] >> C_;
		node["coef"] >> coef_;
		node["gamma"] >> gamma_;
		node["nu"] >> nu_;
		node["degree"] >> degree_;
		node["p"] >> p_;
		node["kernelType"] >> kernelType;
		node["auto_train"] >> trainAuto_;
		node["iteration_count"] >> iteration_count;
		node["hardNegativeMining"] >> hardNegMining_;
		node["epsilon"] >> epsilon;

	
		//default
		model_->setC(1.0e-1);
		model_->setType(SVM::C_SVC);
		model_->setKernel(SVM::LINEAR);
		model_->setTermCriteria (cvTermCriteria(cv::TermCriteria::MAX_ITER || cv::TermCriteria::EPS, 1.0e3, 1.0e-3));


		if (svm_type == "C_SVC"){
			model_->setType(SVM::C_SVC);
		}
		else if (svm_type == "NU_SVC"){
			model_->setType(SVM::NU_SVC);
		}
		else if (svm_type == "ONE_CLASS"){
			model_->setType(SVM::ONE_CLASS);
		}
		else if (svm_type == "EPS_SVR"){
			model_->setType(SVM::EPS_SVR);
		}
		else if (svm_type == "NU_SVR"){
			model_->setType(SVM::NU_SVR);
		}
		else{
			model_->setType(SVM::C_SVC);
		}

		if (kernelType == "LINEAR" || kernelType == "Linear" || kernelType == "linear"){
			model_->setKernel(SVM::LINEAR);
		}
		else if (kernelType == "POLYNOMIAL" || kernelType == "Polynomial" || kernelType == "polynomial" || kernelType == "POLY"){
			model_->setKernel(SVM::POLY);
		}
		else if (kernelType == "RBF" || kernelType == "rbf"){
			model_->setKernel(SVM::RBF);
		}
		else if (kernelType == "SIGMOID" || kernelType == "Sigmoid" || kernelType == "sigmoid"){
			model_->setKernel(SVM::SIGMOID);
		}
		else{
			model_->setKernel(SVM::LINEAR);
		}

		int termCrit = 0;
		if (iteration_count > 0){
			termCrit = cv::TermCriteria::COUNT;
		}
		if (epsilon > 0){
			termCrit += cv::TermCriteria::EPS;
		}
	
		model_->setC(C_);
		model_->setCoef0(coef_);
		model_->setDegree(degree_);
		model_->setGamma(gamma_);
		model_->setNu(nu_);
		model_->setP(p_);
		model_->setTermCriteria(cvTermCriteria(termCrit, iteration_count, epsilon));
	}

	// Classify samples and set responses.
	void SVM_Multiclass::predict(const cv::Mat &X, cv::Mat &responses) {
		responses.release();
		responses.create(cv::Size(static_cast<int>(labels.size()), X.rows), CV_32F);

		for (int row = 0; row < X.rows; row++){
			int label = static_cast<int>(model_->predict(X.row(row)));
			responses.at<float>(row, 0) = static_cast<float>(label);

			float resp = model_->predict(X.row(row), cv::noArray(), StatModel::Flags::RAW_OUTPUT); //VERIFICAR ESSE NOARRAY
			responses.at<float>(row, 1) = resp;
		}
	}

	void SVM_Multiclass::trainSVM(const cv::Mat &samples, const std::vector<std::string> &labelVec){
		double min, max;
		cv::minMaxIdx(samples, &min, &max);
		if (min < -1 && max > 1) std::cout << "libsvm has better results with data in the [-1,1] or [0, 1] range" << std::endl;
		std::vector<std::string> labelsName = retrieveClassIDs();
		cv::Mat_<int> labels;
		cv::Mat_<float> class_weights;
		class_weights.create(static_cast<int>(labelsName.size()), 1); // cria com o número de labels
		class_weights = 0;
		labels.create(samples.rows, 1);
		labels = 0;
		for (int row = 0; row < labelVec.size(); row++){

			for (int i = 0; i < labelsName.size(); i++)
			{
				if (labelVec[row] == labelsName[i]){
					labels[row][0] = i;
					class_weights[i][0]++;
				}
			}
		}
		//class_weights.create(labelsName.size()
		for (int i = 0; i < labelsName.size(); i++)
			class_weights[i][0] /= dataY.size();

		if (trainAuto_){
			model_->setClassWeights(class_weights);
			cv::Ptr<TrainData> trainData = TrainData::create(samples, cv::ml::ROW_SAMPLE, labels);
			model_->trainAuto(trainData);

			std::ofstream fileStream;
			fileStream.open("optimalParamsSVM.txt");
			fileStream << "C," << model_->getC() << "\n";
			fileStream << "Coef," << model_->getCoef0() << "\n";
			fileStream << "degree," << model_->getDegree() << "\n";
			fileStream << "gamma," << model_->getGamma() << "\n";
			fileStream << "kernel," << model_->getKernelType() << "\n";
			fileStream << "nu," << model_->getNu() << "\n";
			fileStream << "p," << model_->getP() << "\n";
			fileStream << "svmtype," << model_->getType() << "\n";
			fileStream << "term_crit," << model_->getTermCriteria().type << "\n";
			fileStream << "term_crit_eps," << model_->getTermCriteria().epsilon << "\n";
			fileStream << "term_crit_it," << model_->getTermCriteria().maxCount << "\n";
			cv::Mat_<float> weights = cv::Mat(model_->getClassWeights());			
			for (int i = 0; i < weights.rows; i++)
				fileStream << "label" << i << "weight," << weights[i][0] << "\n";
			fileStream.close();
		}
		else{
			model_->setClassWeights(class_weights);
			cv::Ptr<TrainData> trainData = TrainData::create(samples, cv::ml::ROW_SAMPLE, labels);
			model_->train(trainData);
		}
	}

	// learn the classifier
	void SVM_Multiclass::learn() {

		cv::Mat samples = dataX;
		std::vector<std::string> labelVec = dataY;
		trainSVM(samples, labelVec);
	}

	// save classifier 
	void SVM_Multiclass::save(cv::FileStorage &storage) {
		if (storage.isOpened() == false){
			std::cerr << "Invalid file storage!";
		}
		//storage << model_;
		storage << "SVM" << "{";
		model_->write(storage); //model_->write(*storage, "SVM");
		storage << "}";
	}

	// load classifier
	void SVM_Multiclass::load(const cv::FileNode &node, cv::FileStorage &storage)  {

		cv::FileNode n1 = node["Labels"];
		cv::FileNodeIterator it = n1.begin(), it_end = n1.end();

		// iterate through a sequence using FileNodeIterator
		for (; it != it_end; ++it)
			labels.insert((std::string)(*it));

		model_ = SVM::create();
		cv::FileNode n = node["SVM"];;
		model_->read(n);
	}

	Classification* SVM_Multiclass::duplicateParameters(){
		SVM_Multiclass* newSVM = new SVM_Multiclass(params);
		return newSVM;
	}
	
	/*
	float RecompSVM::predict(cv::InputArray _samples, cv::OutputArray _results, int flags) const
	{
		float result = 0;
		cv::Mat samples = _samples.getMat(), results;
		int nsamples = samples.rows;
		bool returnDFVal = (flags & RAW_OUTPUT) != 0;
		int var_count = getVarCount();

		CV_Assert(samples.cols == var_count && samples.type() == CV_32F);

		if (_results.needed())
		{
			_results.create(nsamples, 1, samples.type());
			results = _results.getMat();
		}
		else
		{
			CV_Assert(nsamples == 1);
			results = cv::Mat(1, 1, CV_32F, &result);
		}

		PredictBody invoker(this, samples, results, returnDFVal);
		if (nsamples < 10)
			invoker(cv::Range(0, nsamples));
		else
			parallel_for_(cv::Range(0, nsamples), invoker);
		return result;
	}
	*/

}