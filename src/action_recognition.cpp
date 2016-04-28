#include "action_recognition.hpp"

namespace ccr
{
	ActionRecognition::ActionRecognition(std::string paramsPath) {
		//numExtractFeatures = 0;
		this->params.open(paramsPath, cv::FileStorage::READ);
		beforeProcess();
	}

	ActionRecognition::~ActionRecognition() {
		//clearNoLongerUseful();
		params.release();
		featuresProperties.clear();

		if (this->desc != NULL)
		{
			delete this->desc;
			desc = NULL;
		}
		if (this->dict != NULL)
		{
			delete this->dict;
			dict = NULL;
		}
		/*
		if (this->classifier != NULL)
		{
			delete this->classifier;
			classifier = NULL;
		}
		*/
		deleteClassifiers();
	}

	void ActionRecognition::deleteClassifiers()
	{
		for (auto &c : classifiers)
		{
			if (c != NULL)
			{
				delete c;
				c = NULL;
			}
		}
		classifiers.clear();
	}

	void ActionRecognition::clearNoLongerUseful()
	{
		std::string rmdir, outputFold;

		params["featureOutput"] >> outputFold;
		rmdir = "del /f/s/q " + outputFold + " > nul";
		system(rmdir.c_str());

		rmdir = "RMDIR /S /Q ";
		system((rmdir + outputFold).c_str());
	}

	void ActionRecognition::beforeProcess() {
		int cP, cT;
		cv::FileNode node;
		std::string mkdir, outputFold;
		params["videosFile"] >> videosYMLPath;
		params["outputFile"] >> outputFile;
		params["saveBinaryFile"] >> saveBinaryFile;
		cP = static_cast<int>(params["classificationProtocol"]);
		cT = static_cast<int>(params["classificationType"]);

		switch (cP)
		{
		case ClassificationProtocol::Train:
			classificationProtocol = ClassificationProtocol::Train;
			break;
		case ClassificationProtocol::Test:
			classificationProtocol = ClassificationProtocol::Test;
			break;
		case ClassificationProtocol::LeaveOneOut:
			classificationProtocol = ClassificationProtocol::LeaveOneOut;
			break;
		default:
			classificationProtocol = ClassificationProtocol::Train;
		}

		switch (cT)
		{
		case ClassificationType::OneAgainstOne:
			classificationType = ClassificationType::OneAgainstOne;
			break;
		case ClassificationType::OneAgainstAll:
			classificationType = ClassificationType::OneAgainstAll;
			break;
		case ClassificationType::OneAgainstAllMultiLabel:
			classificationType = OneAgainstAllMultiLabel;
			break;
		default:
			classificationType = ClassificationType::OneAgainstOne;
		}


		mkdir = "MKDIR ";
		params["featureOutput"] >> outputFold;
		system((mkdir + outputFold).c_str());
		params["bowOutput"] >> outputFold;
		if (outputFold != "")
			system((mkdir + outputFold).c_str());

		node = params["samplingSetup"]["blocks"];
		for (int i = 0; i < node.size(); ++i) {
			samplingSetup.sampleX = (int)node[i][0]; samplingSetup.sampleY = (int)node[i][1]; samplingSetup.sampleL = (int)node[i][2];
			samplingSetup.strideX = (int)node[i][3]; samplingSetup.strideY = (int)node[i][4]; samplingSetup.strideL = (int)node[i][5];
		}


		node = params["featureParams"];
		int nBMag = node["nBinsMagnitude"];
		int nBAng = node["nBinsAngle"];
		int distMag = node["distanceMagnitude"];
		int distAng = node["distanceAngle"];
		int cubL = node["cuboidLength"];
		float maxMag = node["maxMagnitude"];
		int logQ = node["logQuantization"];
		int movF = node["movementFilter"];
		std::string tempS; node["temporalScales"] >> tempS;
		std::vector<int> vec = splitTemporalScales(tempS, ',');
		int aux = node["extractionType"];
		ssig::ExtractionType extType = static_cast<ssig::ExtractionType>(aux);

		desc = new ssig::OFCM(nBMag, nBAng, distMag, distAng, cubL, maxMag, logQ, static_cast<bool>(movF), vec, extType);


		node = params["visualDictionayParams"];
		int nCWs = node["nCWs"];
		int method = node["method"];
		VisualDictionaryMethod vdMethod;

		switch (method)
		{
		case VisualDictionaryMethod::Random:
			vdMethod = VisualDictionaryMethod::Random;
			break;
		case VisualDictionaryMethod::Kmeans:
			vdMethod = VisualDictionaryMethod::Kmeans;
			break;
		default:
			vdMethod = VisualDictionaryMethod::Random;
		}

		dict = new ccr::VisualDictionary(nCWs, vdMethod);
		//classifier = new ccr::SVM_Multiclass(params);
		classifiers.push_back(new ccr::SVM_Multiclass(params));
	}

	void ActionRecognition::execute()
	{

		switch (classificationProtocol)
		{
		case ClassificationProtocol::Train:
			extractFeatures();
			createDictionary2();
			////loadDictionary();
			extractBagOfWords();
			learnClassificationModel();
			break;

		case ClassificationProtocol::Test:
			extractFeatures();
			loadDictionary();
			extractBagOfWords();
			////loadBagOfWords(); ////
			loadClassifierModel();
			classification();
			break;

		case ClassificationProtocol::LeaveOneOut:
			extractFeatures();
			createDictionary();
			extractBagOfWords();
			leaveOneOut();
			break;

		}
	}

	void ActionRecognition::extractFeatures()
	{
		cv::Mat output;
		std::string videoName, label, path;
		cv::FileNode node;
		cv::FileNodeIterator inode;
		cv::FileStorage videosStorage; // storageNumFeat;

		videosStorage.open(videosYMLPath, cv::FileStorage::READ);
		node = videosStorage["videos"];

		for (inode = node.begin(); inode != node.end(); ++inode) {
			videoName = (std::string) (*inode)["dir"];
			std::vector<std::string> split = splitString(videoName, '/');
			videoName = splitString(split[split.size() - 1], '.')[0];
			std::cout << "Extracting features from " << videoName;

			video.clear();

			loadVideoFrames(inode);
			label = (*inode)["objID"].isNone() ? "" : (std::string)(*inode)["objID"];
			std::cout << " .";

			cuboids.clear();
			createCuboids();
			std::cout << ".";

			output.release();
			desc->release();
			desc->setData(video);
			video.clear(); //not used anymore
			desc->extract(cuboids, output);
			std::cout << ".";
			//numExtractFeatures += output.rows;

			FeatureIndex fi(getFileName(videoName, params), label, output.rows, output.cols, 0); // 0 is  a dummy number
			featuresProperties.push_back(fi);

			desc->release();

			if (saveBinaryFile)
				saveFeatureBinary(label, output, videoName);
			else
				saveFeature(label, output, videoName);

			std::cout << " OK" << std::endl;
		}
		/*
		params["featureOutput"] >> outputFold;

		if (classificationProtocol == ClassificationProtocol::Train)
		protocol = "Train";
		else if (classificationProtocol == ClassificationProtocol::Test)
		protocol = "Test";

		path = outputFold + "\\numFeatures" + protocol + ".yml";
		storageNumFeat.open(path, cv::FileStorage::WRITE);
		if (storageNumFeat.isOpened() == false)
		std::cerr << "Invalid file storage!" << std::endl;
		storageNumFeat << "Features" << numExtractFeatures;

		storageNumFeat.release();
		*/
		output.release();
		videosStorage.release();
		video.clear();
		cuboids.clear();
		delete desc;
		desc = NULL;
	}

	void ActionRecognition::createDictionary()
	{
		cv::FileStorage storage;
		std::string dictModelFile;

		std::cout << "\n\nBuilding dictionary ";

		params["dictionaryFile"] >> dictModelFile;

		// Open modelFile
		storage.open(dictModelFile, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;

		storage << "ActionRecognitionDictionary" << "{";

		std::cout << ".";

		if (featuresProperties.size() == 0)
			fillFeaturesProperties();

		for (auto &p : featuresProperties)
			dict->addFeatureVectors(p);

		std::cout << ".";

		// Create dictionary 
		dict->buildDictionary();
		std::cout << ".";

		// Save dictionary
		dict->save(storage);
		std::cout << " OK " << std::endl << std::endl;

		//featuresPath.clear();
		storage.release();
	}

	void ActionRecognition::createDictionary2()
	{
		std::string path;
		cv::FileStorage storage;
		std::string dictModelFile;

		std::cout << "\n\nBuilding dictionary ";

		params["dictionaryFile"] >> dictModelFile;

		// Open modelFile
		storage.open(dictModelFile, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;

		storage << "ActionRecognitionDictionary" << "{";

		std::cout << ".";

		params["featureOutput"] >> path;

		std::cout << ".";

		// Create dictionary 
		dict->buildDictionary2(path);
		if (featuresProperties.size() == 0)
			featuresProperties = dict->getFeaturesProperties();
		dict->clearFeaturesProperties();
		std::cout << ".";

		// Save dictionary
		dict->save(storage);
		std::cout << " OK " << std::endl << std::endl;

		//featuresPath.clear();
		storage.release();
	}

	void ActionRecognition::extractBagOfWords()
	{
		int numVideos = 0;
		int v = 0;
		std::vector<std::thread> threads;

		if (featuresProperties.size() == 0)
			fillFeaturesProperties();

		numVideos = static_cast<int>(featuresProperties.size());

		while (v < numVideos)
		{
			std::cout << "Extracting Bag of Words for videos ";
			std::vector< std::pair<cv::Mat, std::string> > bagOfWordsVector;
			std::map<std::string, std::string> mapPathToLabel;
			for (int cores = 0; cores < std::thread::hardware_concurrency() && v < numVideos; cores++, v++)
			{
				cv::Mat bow;
				bow.release();
				bow.create(1, dict->getnCWs(), CV_32F);
				bow = 0;
				std::pair<cv::Mat, std::string> p(bow, featuresProperties[v].path);
				bagOfWordsVector.push_back(p);
				mapPathToLabel[p.second] = featuresProperties[v].label;
				std::cout << v << ", ";
			}
			std::cout << ".";
			for (auto& p : bagOfWordsVector)
				threads.push_back(std::thread(&ActionRecognition::createBoW, this, p.first, p.second));

			std::cout << ".";
			for (auto& th : threads)
				th.join();

			std::cout << ".";
			//for (auto& p : bagOfWordsVector) // Old way. Used just for one label per video
			//this->mapLabelToBoW[mapPathToLabel[p.second]].push_back(p.first); //Used just for one label per video

			for (auto& p : bagOfWordsVector) // New way. Used just for multi-label videos
			{
				std::vector<std::string> multiLabel = splitString(mapPathToLabel[p.second], ',');
				for (std::string &label : multiLabel)
				{
					this->mapLabelToBoW[label].push_back(p.first);
					std::vector<std::string> videoName = splitString(p.second, '\\');
					videoName = splitString(videoName[videoName.size() - 1], '.');
					saveBoW(label, p.first, videoName[0]);
				}
			}

			std::cout << " OK" << std::endl;
			threads.clear();
		}
	}

	void ActionRecognition::loadBagOfWords()
	{
		DIR *dir = 0;
		std::string path;
		std::string label;
		cv::FileNode node, n1;
		cv::Mat feature;

		params["bowOutput"] >> path;
		dir = opendir(path.c_str());

		if (dir == 0) {
			std::cerr << "Impossible to open directory." << std::endl;
			exit(1);
		}

		mapLabelToBoW.clear();

		struct dirent *featFile = 0;
		while (featFile = readdir(dir))
		{
			std::string file = featFile->d_name;
			if (featFile->d_type == isFile)
			{
				cv::FileStorage storageFeat;
				std::string featurePath = path + "\\" + file;

				// binary file
				if (featurePath[featurePath.size() - 1] == 'n')
				{
					feature = matRead(featurePath, label);
				}
				else
				{
					//Loading feature
					storageFeat.open(path + "\\" + file, cv::FileStorage::READ);
					if (storageFeat.isOpened() == false)
						std::cerr << "Invalid file storage " << (path + file + "!") << std::endl;

					node = storageFeat.root();
					n1 = node["ActionRecognitionFeatures"];
					n1["Label"] >> label;
					n1["Features"] >> feature;
					storageFeat.release();
				}

				this->mapLabelToBoW[label].push_back(feature.clone());
			}
		}
	}

	void ActionRecognition::createBoW(cv::Mat bagOfWords, std::string featurePath)
	{
		cv::Mat feature;
		cv::FileStorage storageFeat;
		cv::FileNode node, n1;
		std::string dummy;

		// binary file
		if (featurePath[featurePath.size() - 1] == 'n')
		{
			feature = matRead(featurePath, dummy);
		}
		else
		{
			//Loading feature
			storageFeat.open(featurePath, cv::FileStorage::READ);
			if (storageFeat.isOpened() == false)
				std::cerr << "Invalid file storage " << (featurePath + "!") << std::endl;

			node = storageFeat.root();
			n1 = node["ActionRecognitionFeatures"];
			n1["Features"] >> feature;
			storageFeat.release();
		}

		for (int r = 0; r < feature.rows; r++) //hard assignment, ModObjectDetectionBOW parece fazer soft assignment...
		{
			cv::Mat partialBagging;
			dict->computeBag(feature.row(r), partialBagging);
			int idx = static_cast<int>(partialBagging.at<float>(1, 0)); //[0][0] contains the distance, [0][1] contains the index
			bagOfWords.at<float>(0, idx) += 1;
		}
		cv::normalize(bagOfWords, bagOfWords, 1, cv::NORM_L2);
	}

	cv::Mat ActionRecognition::matRead(const std::string& filename, std::string &label)
	{
		std::ifstream fs(filename, std::fstream::binary);

		// Header
		char* temp;
		int size, rows, cols, type, channels;

		fs.read((char*)&size, sizeof(int));         // label size
		temp = new char[size + 1];
		fs.read(temp, size);												// label
		temp[size] = '\0';
		label = temp;
		delete[] temp;

		fs.read((char*)&rows, sizeof(int));         // rows
		fs.read((char*)&cols, sizeof(int));         // cols
		fs.read((char*)&type, sizeof(int));         // type
		fs.read((char*)&channels, sizeof(int));     // channels

		// Data
		cv::Mat mat(rows, cols, type);
		fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

		return mat;
	}

	void ActionRecognition::learnClassificationModel()
	{
		if (this->classificationType == ClassificationType::OneAgainstOne)
			learnOneAgainstOneClassification();
		else if (this->classificationType == ClassificationType::OneAgainstAll || this->classificationType == ClassificationType::OneAgainstAllMultiLabel)
			learnOneAgainstAllClassification();
	}

	void ActionRecognition::learnOneAgainstOneClassification()
	{
		std::string classModelFile;
		cv::FileStorage storage;
		std::vector<std::string> labelsName;

		params["classModelFile"] >> classModelFile;

		// Open modelFile
		storage.open(classModelFile, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage " << (classModelFile + "!") << std::endl;

		storage << "ActionRecognitionClassificationModel" << "{";

		for (std::pair<std::string, std::vector<cv::Mat>> p : mapLabelToBoW)
			for (cv::Mat trainData : p.second)
				classifiers[0]->addSamples(trainData, p.first);

		classifiers[0]->learn();
		labelsName = classifiers[0]->retrieveClassIDs();
		storage << "Labels" << labelsName;
		classifiers[0]->save(storage);
		storage.release();
	}

	void ActionRecognition::learnOneAgainstAllClassification()
	{
		std::string split, classModelFile;
		cv::FileStorage storage;
		std::vector<std::string> labelsName, splitStr;

		//params["classModelFile"] >> classModelFile;
		params["classModelFile"] >> split;
		splitStr = splitString(split, '.');

		int i = 0;
		for (std::map<std::string, std::vector<cv::Mat>>::iterator it = mapLabelToBoW.begin(); it != mapLabelToBoW.end(); ++it)
		{
			std::string oneLabel = it->first;
			std::ostringstream intConvert;
			intConvert << i;
			classModelFile = splitStr[0] + intConvert.str() + ".yml";
			i++;

			// Open modelFile
			storage.open(classModelFile, cv::FileStorage::WRITE);
			if (storage.isOpened() == false)
				std::cerr << "Invalid file storage " << (classModelFile + "!") << std::endl;

			storage << "ActionRecognitionClassificationModel" << "{";

			classifiers[0]->reset();
			for (std::pair<std::string, std::vector<cv::Mat>> p : mapLabelToBoW)
			{
				for (cv::Mat trainData : p.second)
				{
					if (oneLabel == p.first)
						classifiers[0]->addSamples(trainData, oneLabel); //Insert the "One" label
					else
						classifiers[0]->addSamples(trainData, "Rest"); //Insert the "Rest", the others
				}
			}

			classifiers[0]->learn();
			labelsName = classifiers[0]->retrieveClassIDs();
			storage << "Labels" << labelsName;
			classifiers[0]->save(storage);
			storage.release();
		}
	}

	void ActionRecognition::loadDictionary()
	{
		cv::FileStorage storage;
		cv::FileNode node, n1;
		std::string dictModelFile;

		params["dictionaryFile"] >> dictModelFile;

		//Loading Dictionary
		storage.open(dictModelFile, cv::FileStorage::READ);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage " << dictModelFile << "!" << std::endl;

		node = storage.root();
		n1 = node["ActionRecognitionDictionary"];
		dict->load(n1, storage);
		storage.release();
	}

	void ActionRecognition::loadOneAgainstOneClassifierModel()
	{
		cv::FileStorage storage;
		cv::FileNode node, n1;
		std::string classModelFile;

		params["classModelFile"] >> classModelFile;

		//Loading Classifier Model
		storage.open(classModelFile, cv::FileStorage::READ);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage " << classModelFile << "!" << std::endl;

		node = storage.root();
		n1 = node["ActionRecognitionClassificationModel"];
		classifiers[0]->load(n1, storage);
		storage.release();
	}

	void ActionRecognition::loadOneAgainstAllClassifierModel()
	{
		int numClassifiers = retrieveClassIds().size();
		deleteClassifiers();

		cv::FileStorage storage;
		cv::FileNode node, n1;
		std::string split, classModelFile;
		std::vector<std::string> splitStr;

		params["classModelFile"] >> split;
		splitStr = splitString(split, '.');

		for (int i = 0; i < numClassifiers; i++)
		{
			std::ostringstream intConvert;
			intConvert << i;
			classModelFile = splitStr[0] + intConvert.str() + ".yml";

			//Loading Classifier Model
			storage.open(classModelFile, cv::FileStorage::READ);
			if (storage.isOpened() == false)
				std::cerr << "Invalid file storage " << classModelFile << "!" << std::endl;

			node = storage.root();
			n1 = node["ActionRecognitionClassificationModel"];
			classifiers.push_back(new ccr::SVM_Multiclass(params));
			classifiers[i]->load(n1, storage);
			storage.release();
		}

	}

	void ActionRecognition::loadClassifierModel()
	{
		if (this->classificationType == ClassificationType::OneAgainstOne)
			loadOneAgainstOneClassifierModel();
		else if (this->classificationType == ClassificationType::OneAgainstAll || this->classificationType == ClassificationType::OneAgainstAllMultiLabel)
			loadOneAgainstAllClassifierModel();
	}

	void ActionRecognition::createCuboids()
	{
		cuboids.clear();

		int videoHeight = video[0].rows;
		int videoWidth = video[0].cols;

		for (int t = 0; t <= static_cast<int>(0 + video.size() - samplingSetup.sampleL); t += samplingSetup.strideL)
			for (int y = 0; y <= static_cast<int>(0 + videoHeight - samplingSetup.sampleY); y += samplingSetup.strideY)
				for (int x = 0; x <= static_cast<int>(0 + videoWidth - samplingSetup.sampleX); x += samplingSetup.strideX)
					cuboids.push_back(ssig::Cube(x, y, t, samplingSetup.sampleX, samplingSetup.sampleY, samplingSetup.sampleL));
	}

	void ActionRecognition::loadVideoFrames(cv::FileNodeIterator &inode)
	{
		int step, nSamples = 0;
		unsigned long int videoLength;
		std::string path;
		cv::VideoCapture capture;
		cv::Mat image;

		video.clear();

		// load video
		path = (std::string) (*inode)["dir"];

		// check video loaded
		capture.open(path);
		if (!capture.isOpened())
			std::cerr << "Error processing file. Can't read video " << path << std::endl;

		videoLength = static_cast<unsigned long>(capture.get(CV_CAP_PROP_FRAME_COUNT));
		nSamples = (*inode)["nSamples"].isNone() ? 0 : (int)(*inode)["nSamples"];

		if (nSamples > 0)
			step = (int)(videoLength / nSamples); // generate step based on the number of samples
		else
			step = (*inode)["step"].isNone() ? 1 : (int)(*inode)["step"]; // load step

		for (unsigned long int frameStep = 0; frameStep < videoLength; frameStep += step)
		{
			capture.set(CV_CAP_PROP_POS_FRAMES, frameStep);
			capture.read((image));
			
			if (image.empty())
				std::cerr << "Error processing file. Can't read frame " << frameStep << " from video " << path << std::endl;
			else
				video.push_back(image.clone());
		}

	}

	void ActionRecognition::saveFeature(std::string label, cv::Mat &features, std::string videoName) {

		cv::FileStorage storage;
		std::string path;
		// Open modelFile
		path = getFileName(videoName, params);

		storage.open(path, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;

		storage << "ActionRecognitionFeatures" << "{";
		storage << "Label" << label;
		storage << "Features" << features;

		storage.release();

		//featuresPath.push_back(path);
	}

	void ActionRecognition::saveBoW(std::string label, cv::Mat &features, std::string videoName) {

		std::string path;
		params["bowOutput"] >> path;
		if (path != "")
		{
			cv::FileStorage storage;
			path = path + "\\" + videoName + ".yml";

			// Open modelFile
			storage.open(path, cv::FileStorage::WRITE);
			if (storage.isOpened() == false)
				std::cerr << "Invalid file storage!" << std::endl;

			storage << "ActionRecognitionFeatures" << "{";
			storage << "Label" << label;
			storage << "Features" << features;

			storage.release();

			//featuresPath.push_back(path);
		}
	}

	void ActionRecognition::saveFeatureBinary(std::string label, cv::Mat &features, std::string videoName) {

		std::string  path = getFileName(videoName, params);
		std::ofstream fs(path, std::fstream::binary);

		// Header
		int type = features.type();
		int channels = features.channels();
		int size = label.size();
		fs.write((char*)&size, sizeof(int));						// label size
		fs.write(label.c_str(), label.size());					// label
		fs.write((char*)&features.rows, sizeof(int));		// rows
		fs.write((char*)&features.cols, sizeof(int));		// cols
		fs.write((char*)&type, sizeof(int));						// type
		fs.write((char*)&channels, sizeof(int));				// channels

		// Data
		if (features.isContinuous())
		{
			fs.write(features.ptr<char>(0), (features.dataend - features.datastart));
		}
		else
		{
			int rowsz = CV_ELEM_SIZE(type) * features.cols;
			for (int r = 0; r < features.rows; ++r)
			{
				fs.write(features.ptr<char>(r), rowsz);
			}
		}
		fs.close();
	}

	inline void ActionRecognition::fillFeaturesProperties()
	{
		DIR *dir = 0;
		std::string path;
		int rows;
		unsigned short cols;
		std::string label;
		cv::FileNode node, n1;

		params["featureOutput"] >> path;
		dir = opendir(path.c_str());

		if (dir == 0) {
			std::cerr << "Impossible to open directory." << std::endl;
			exit(1);
		}

		struct dirent *featFile = 0;
		while (featFile = readdir(dir))
		{
			std::string file = featFile->d_name;
			//fazer verificação de AVI e JPG :-)
			if (featFile->d_type == isFile)
			{
				cv::FileStorage storageFeat;
				std::string featurePath = path + "\\" + file;

				// binary file
				if (featurePath[featurePath.size() - 1] == 'n')
				{
					cv::Mat feature = matRead(featurePath, label);
					rows = feature.rows;
					cols = feature.cols;
				}
				else
				{
					//Loading feature
					storageFeat.open(path + "\\" + file, cv::FileStorage::READ);
					if (storageFeat.isOpened() == false)
						std::cerr << "Invalid file storage " << (path + file + "!") << std::endl;

					node = storageFeat.root();
					n1 = node["ActionRecognitionFeatures"];
					n1["Label"] >> label;
					node = n1["Features"];
					node["rows"] >> rows;
					node["cols"] >> cols;
					storageFeat.release();
				}

				FeatureIndex fi((path + "\\" + file), label, rows, cols, 0); // 0 is  a dummy number
				featuresProperties.push_back(fi);
			}
		}
	}

	void ActionRecognition::classification()
	{
		if (this->classificationType == ClassificationType::OneAgainstOne)
			oneAgainstOneClassification();
		else if (this->classificationType == ClassificationType::OneAgainstAll)
			oneAgainstAllClassification();
		else if (this->classificationType == ClassificationType::OneAgainstAllMultiLabel)
			oneAgainstAllMultiLabelClassification();
	}

	void ActionRecognition::oneAgainstOneClassification()
	{
		int nLabels = classifiers[0]->retrieveClassIDs().size();
		cv::Mat_<float> confusionMat(nLabels, nLabels);
		std::vector<float> **confusionMatScores;
		confusionMat = 0;

		confusionMatScores = new std::vector<float>*[nLabels];
		for (int i = 0; i < nLabels; i++)
			confusionMatScores[i] = new std::vector<float>[nLabels];

		std::ofstream class_report;
		class_report.open("class_report.txt", std::ofstream::out | std::ofstream::app);
		class_report << "realClass\tpredictedClass\tresp" << std::endl;

		for (std::pair<std::string, std::vector<cv::Mat> > p : mapLabelToBoW)
		{
			int realClass = classifiers[0]->retrieveResponseClassIDPosition(p.first);
			for (cv::Mat_<float> testData : p.second)
			{
				cv::Mat_<float> responses;
				classifiers[0]->predict(testData, responses);
				int predictedClass = responses[0][0];
				float resp = responses[0][1];
				confusionMat[realClass][predictedClass]++;
				confusionMatScores[realClass][predictedClass].push_back(resp);

				class_report << realClass << "\t" << predictedClass << "\t" << resp << std::endl;
			}
		}

		class_report.close();

		generateOutput(nLabels, confusionMat, confusionMatScores);

		for (int i = 0; i < nLabels; i++)
			delete[] confusionMatScores[i];
		delete[] confusionMatScores;
	}

	void ActionRecognition::oneAgainstAllClassification()
	{
	}

	void ActionRecognition::oneAgainstAllMultiLabelClassification()
	{

		std::vector<cv::Mat_<float>> vecConfusionMat;
		std::vector<std::vector<float>**> vecConfusionMatScores;

		for (int c = 0; c < classifiers.size(); c++)
		{
			std::vector<std::string> ids = classifiers[c]->retrieveClassIDs();
			int nLabels = ids.size();
			cv::Mat_<float> confusionMat(nLabels, nLabels);
			std::vector<float> **confusionMatScores;
			confusionMat = 0;

			confusionMatScores = new std::vector<float>*[nLabels];
			for (int i = 0; i < nLabels; i++)
				confusionMatScores[i] = new std::vector<float>[nLabels];

			std::ofstream class_report;
			std::ostringstream intConvert;
			intConvert << c;
			class_report.open("class_report" + intConvert.str() + ".txt", std::ofstream::out | std::ofstream::app);
			class_report << "realClass\tpredictedClass\tresp" << std::endl;

			for (std::pair<std::string, std::vector<cv::Mat> > p : mapLabelToBoW)
			{
				std::string idLabel;
				if (p.first == ids[0])
					idLabel = p.first;
				else
					idLabel = ids[1];

				int realClass = classifiers[c]->retrieveResponseClassIDPosition(idLabel);
				for (cv::Mat_<float> testData : p.second)
				{
					cv::Mat_<float> responses;
					classifiers[c]->predict(testData, responses);
					int predictedClass = responses[0][0];
					float resp = responses[0][1];
					confusionMat[realClass][predictedClass]++;
					confusionMatScores[realClass][predictedClass].push_back(resp);

					class_report << realClass << "\t" << predictedClass << "\t" << resp << std::endl;
				}
			}
			vecConfusionMat.push_back(confusionMat);
			vecConfusionMatScores.push_back(confusionMatScores);
			class_report.close();
		}

		generateOutputMultiLabel(vecConfusionMat, vecConfusionMatScores);

		for (auto &confusionMatScores : vecConfusionMatScores)
		{
			for (int i = 0; i < classifiers[0]->retrieveClassIDs().size(); i++)
				delete[] confusionMatScores[i];
			delete[] confusionMatScores;
		}
	}

	void ActionRecognition::leaveOneOut()
	{
		int i = 0;
		int numVideos, percent;
		std::vector<std::thread> threads;
		std::vector<ccr::Classification> classifiers;
		int nLabels = mapLabelToBoW.size();
		cv::Mat_<float> confusionMat(nLabels, nLabels);
		std::vector<float> **confusionMatScores;
		confusionMat = 0;

		std::cout << "\n\nLeave One Out ";

		confusionMatScores = new std::vector<float>*[nLabels];
		for (int j = 0; j < nLabels; j++)
			confusionMatScores[j] = new std::vector<float>[nLabels];

		std::ofstream class_report;
		class_report.open("class_report.txt", std::ofstream::out | std::ofstream::trunc); //std::ofstream::app
		class_report << "realClass\tpredictedClass\tresp" << std::endl;
		std::cout << ".";

		if (featuresProperties.size() == 0)
			fillFeaturesProperties();

		std::cout << ". ";

		numVideos = static_cast<int>(featuresProperties.size());
		
		while (i < numVideos)
		{
			percent = (i * 100) / numVideos;
			std::cout << percent << "%";
			std::vector<ClassifierResponse> classifierResponseVector;

			for (int cores = 0; cores < std::thread::hardware_concurrency() && i < numVideos; cores++, i++)
			{
				ClassifierResponse cr = { i, 0, 0, 0.0 };
				classifierResponseVector.push_back(cr);
			}

			for (auto& cr : classifierResponseVector)
				threads.push_back(std::thread(&ActionRecognition::classificationThread, this, &cr));

			for (auto& th : threads)
				th.join();

			for (auto& cr : classifierResponseVector)
			{
				confusionMat[cr.realClass][cr.predictedClass]++;
				confusionMatScores[cr.realClass][cr.predictedClass].push_back(cr.resp);
				class_report << cr.realClass << "\t" << cr.predictedClass << "\t" << cr.resp << std::endl;
			}

			threads.clear();

			if (percent > 9)
				std::cout << "\b\b\b";
			else
				std::cout << "\b\b";
		}

		std::cout << "100%";
		std::cout << "\b\b\b\b\b";
		class_report.close();

		generateOutput(nLabels, confusionMat, confusionMatScores);
		std::cout << ".";

		for (int i = 0; i < nLabels; i++)
			delete[] confusionMatScores[i];
		delete[] confusionMatScores;

		std::cout << " OK " << std::endl << std::endl;;
	}

	void ActionRecognition::classificationThread(ClassifierResponse *cr)
	{
		int j = 0;
		cv::Mat_<float> testData;
		std::string  testLabel;

		ccr::Classification* c;
		classifiers[0]->reset();
		c = classifiers[0]->duplicateParameters();
		c->reset();

		for (std::pair<std::string, std::vector<cv::Mat> > p : mapLabelToBoW)
		{
			for (cv::Mat_<float> trainData : p.second)
			{
				if (cr->i == j)
				{
					testData = trainData;
					testLabel = p.first;
					j++;
				}
				else
				{
					c->addSamples(trainData, p.first);
					j++;
				}
			}
		}

		c->learn();

		cr->realClass = c->retrieveResponseClassIDPosition(testLabel);
		cv::Mat_<float> responses;
		c->predict(testData, responses);
		cr->predictedClass = responses[0][0];
		cr->resp = responses[0][1];

		delete c;
	}

	inline void ActionRecognition::generateOutput(int nLabels, cv::Mat_<float> confusionMat, std::vector<float> **confusionMatScores)
	{
		cv::FileStorage storage;
		cv::Mat_<float> output;
		output.release();
		output.create(cv::Size(9, nLabels)); //4 = TP, FP, FN, TN, precision, recall, specificity, accuracy per class, balanced accuracy per class 
		output = 0;
		float precision, recall, specificity;
		std::map<int, std::vector<float>> TPScores, FPScores, FNScores, TNScores; //first = class, second = scores

		//Create output file
		storage.open(outputFile, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;

		storage << "ActionRecognitionClassificationOutput" << "{";

		// TP - True Positive
		for (int tp = 0; tp < nLabels; tp++)
		{
			output[tp][0] = confusionMat[tp][tp];
			for (int score = 0; score < confusionMatScores[tp][tp].size(); score++)
				TPScores[tp].push_back(confusionMatScores[tp][tp].at(score));
		}

		// FP - False Positive
		for (int col = 0; col < nLabels; col++)
		{
			for (int row = 0; row < nLabels; row++)
			{
				output[col][1] += confusionMat[row][col];

				if (col != row)
					for (int score = 0; score < confusionMatScores[row][col].size(); score++)
						FPScores[col].push_back(confusionMatScores[row][col].at(score));
			}
			output[col][1] -= output[col][0];
		}

		// FN - False Negative
		for (int row = 0; row < nLabels; row++)
		{
			for (int col = 0; col < nLabels; col++)
			{
				output[row][2] += confusionMat[row][col];

				if (col != row)
					for (int score = 0; score < confusionMatScores[row][col].size(); score++)
						FNScores[row].push_back(confusionMatScores[row][col].at(score));
			}
			output[row][2] -= output[row][0];
		}

		// TN - True Negative
		cv::Scalar sum = cv::sum(confusionMat);
		for (int tn = 0; tn < nLabels; tn++)
			output[tn][3] = sum[0] - output[tn][0] - output[tn][1] - output[tn][2];

		for (int l = 0; l < nLabels; l++)
		{
			precision = output[l][0] / (output[l][0] + output[l][1]);
			output[l][4] = precision;
			recall = output[l][0] / (output[l][0] + output[l][2]);
			output[l][5] = recall;
			specificity = output[l][3] / (output[l][3] + output[l][1]);
			output[l][6] = specificity;
			output[l][7] = (output[l][0] + output[l][3]) / (output[l][0] + output[l][1] + output[l][2] + output[l][3]); //accuracy per class
			output[l][8] = balancedAccuracy(output[l][0], output[l][1], output[l][2], output[l][3]); //balanced accuracy per class
		}

		float *ap = new float[nLabels];
		for (int label = 0; label < nLabels; label++)
			ap[label] = averagePrecision(label, output[label][0], output[label][2], TPScores, FPScores);

		double meanAP = std::accumulate(ap, ap + nLabels, 0.0f);
		double meanACC = meanAccuracy(output.col(7));
		double stdDev = stdDeviation(output.col(7), meanACC);
		double meanBalACC = meanAccuracy(output.col(8));
		double stdDevBal = stdDeviation(output.col(8), meanBalACC);

		storage << "MeanBalancedAccuracy" << meanBalACC;
		storage << "StandardDeviationBalACC" << stdDevBal;
		storage << "MeanAccuracy" << meanACC;
		storage << "StandardDeviationACC" << stdDev;
		storage << "MeanAveragePrecision" << meanAP;

		std::vector<std::string> labelsName = retrieveClassIds();
		for (int label = 0; label < nLabels; label++)
		{
			std::stringstream classLabel;
			classLabel << "Class" << label;
			storage << classLabel.str() << "{";
			storage << "Label" << labelsName[label];
			storage << "TP" << output[label][0];
			storage << "FP" << output[label][1];
			storage << "FN" << output[label][2];
			storage << "TN" << output[label][3];
			storage << "precision" << output[label][4];
			storage << "recall" << output[label][5];
			storage << "specificity" << output[label][6];
			storage << "accPerClass" << output[label][7];
			storage << "accBalPerClass" << output[label][8];
			storage << "apPerClass" << ap[label];
			storage << "}";
		}
		storage.release();

		delete ap;
	}

	void ActionRecognition::generateOutputMultiLabel(std::vector<cv::Mat_<float>> vecConfusionMat, std::vector<std::vector<float>**> vecConfusionMatScores)
	{
		std::vector<cv::Mat_<float>> vecOutput;		
		cv::FileStorage storage;
		double meanAP = 0.0;
		double meanACC = 0.0;
		double stdDev = 0.0;
		int numClassifiers = classifiers.size();
		float *vecAp = new float[numClassifiers];

		//Create output file
		storage.open(outputFile, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;

		storage << "ActionRecognitionClassificationOutput" << "{";
		for (int c = 0; c < numClassifiers; c++)
		{
			cv::Mat_<float> confusionMat = vecConfusionMat[c];
			std::vector<float>** confusionMatScores = vecConfusionMatScores[c];
			cv::Mat_<float> output;
			int nLabels = classifiers[c]->retrieveClassIDs().size();

			output.release();
			output.create(cv::Size(8, nLabels)); //4 = TP, FP, FN, TN, precision, recall, specificity, accuracy per class
			output = 0;
			float precision, recall, specificity;
			std::map<int, std::vector<float>> TPScores, FPScores, FNScores, TNScores; //first = class, second = scores

			// TP - True Positive
			for (int tp = 0; tp < nLabels; tp++)
			{
				output[tp][0] = confusionMat[tp][tp];
				for (int score = 0; score < confusionMatScores[tp][tp].size(); score++)
					TPScores[tp].push_back(confusionMatScores[tp][tp].at(score));
			}

			// FP - False Positive
			for (int col = 0; col < nLabels; col++)
			{
				for (int row = 0; row < nLabels; row++)
				{
					output[col][1] += confusionMat[row][col];

					if (col != row)
						for (int score = 0; score < confusionMatScores[row][col].size(); score++)
							FPScores[col].push_back(confusionMatScores[row][col].at(score));
				}
				output[col][1] -= output[col][0];
			}

			// FN - False Negative
			for (int row = 0; row < nLabels; row++)
			{
				for (int col = 0; col < nLabels; col++)
				{
					output[row][2] += confusionMat[row][col];

					if (col != row)
						for (int score = 0; score < confusionMatScores[row][col].size(); score++)
							FNScores[row].push_back(confusionMatScores[row][col].at(score));
				}
				output[row][2] -= output[row][0];
			}

			// TN - True Negative
			cv::Scalar sum = cv::sum(confusionMat);
			for (int tn = 0; tn < nLabels; tn++)
				output[tn][3] = sum[0] - output[tn][0] - output[tn][1] - output[tn][2];

			for (int l = 0; l < nLabels; l++)
			{
				precision = output[l][0] / (output[l][0] + output[l][1]);
				output[l][4] = precision;
				recall = output[l][0] / (output[l][0] + output[l][2]);
				output[l][5] = recall;
				specificity = output[l][3] / (output[l][3] + output[l][1]);
				output[l][6] = specificity;
				output[l][7] = (output[l][0] + output[l][3]) / (output[l][0] + output[l][1] + output[l][2] + output[l][3]); //accuracy per class
			}

			vecAp[c] = averagePrecision(0, output[0][0], output[0][2], TPScores, FPScores);
			
			vecOutput.push_back(output);
		}

		meanAP = std::accumulate(vecAp, vecAp + numClassifiers, 0.0f);
		meanACC = meanAccuracyOneAgainstAll(vecOutput);
		stdDev = stdDeviationOneAgainstAll(vecOutput, meanACC);

		storage << "MeanAccuracy" << meanACC;
		storage << "StandardDeviationACC" << stdDev;
		storage << "MeanAveragePrecision" << meanAP;

		std::vector<std::string> labelsName = retrieveLabelsNameOneAgainstAll();
		for (int label = 0; label < numClassifiers; label++)
		{
			cv::Mat_<float> output = vecOutput[label];
			std::stringstream classLabel;
			classLabel << "Class" << label;
			storage << classLabel.str() << "{";
			storage << "Label" << labelsName[label];
			storage << "TP" << output[0][0];
			storage << "FP" << output[0][1];
			storage << "FN" << output[0][2];
			storage << "TN" << output[0][3];
			storage << "precision" << output[0][4];
			storage << "recall" << output[0][5];
			storage << "specificity" << output[0][6];
			storage << "accPerClass" << output[0][7];
			storage << "apPerClass" << vecAp[label];
			storage << "}";
		}
		storage.release();

		delete vecAp;
	}

	std::vector<std::string> ActionRecognition::retrieveClassIds()
	{
		std::vector<std::string> ids;
		//classifier[0]->reset(); ??????

		for (auto& p : mapLabelToBoW)
			ids.push_back(p.first);

		return ids;
	}

	std::vector<std::string> ActionRecognition::retrieveLabelsNameOneAgainstAll()
	{
		std::vector<std::string> ids;

		for (int i = 0; i < classifiers.size(); i++)
			ids.push_back(classifiers[i]->retrieveClassIDs()[0]);

		return ids;
	}

	double stdDeviation(cv::Mat_<float> list, double mean)
	{
		double sum = 0.0;
		double standDev = 0.0;

		for (int i = 0; i < list.rows; i++)
			sum += (list[i][0] - mean)*(list[i][0] - mean);

		sum = (double)(sum / (list.rows - 1));
		standDev = sqrt(sum);

		return standDev;
	}

	double stdDeviationOneAgainstAll(std::vector<cv::Mat_<float>> list, double mean)
	{
		double sum = 0.0;
		double standDev = 0.0;

		for (auto &output : list)
			sum += (output[0][7] - mean)*(output[0][7] - mean);

		sum = (double)(sum / (list.size() - 1));
		standDev = sqrt(sum);

		return standDev;
	}

	double meanAccuracy(cv::Mat_<float> list)
	{
		double sum = 0.0;
		double mean = 0.0;

		for (int i = 0; i < list.rows; i++)
			sum += list[i][0];

		mean = (double)(sum / (list.rows));

		return mean;
	}

	double meanAccuracyOneAgainstAll(std::vector<cv::Mat_<float>> list)
	{
		double sum = 0.0;
		double mean = 0.0;

		for (auto &output : list)
			sum += output[0][7];

		mean = (double)(sum / (list.size()));

		return mean;
	}

	float balancedAccuracy(int TP, int FP, int FN, int TN)
	{
		float percentTP, percentFP, percentFN, percentTN;

		percentTP = static_cast<float>(TP * 100.0) / static_cast<float>(TP + FN);
		percentFN = 100.0 - percentTP;

		percentFP = static_cast<float>(FP * 100.0) / static_cast<float>(TN + FP);
		percentTN = 100.0 - percentFP;

		return (percentTP + percentTN) / (percentTP + percentTN + percentFN + percentFP);
	}

	float averagePrecision(int label, int numTp, int numFn, std::map<int, std::vector<float>> TPScores, std::map<int, std::vector<float>> FPScores)
	{
		std::vector<int> tpVec, fpVec;

		std::sort(TPScores[label].begin(), TPScores[label].end(), std::greater<float>());
		std::sort(FPScores[label].begin(), FPScores[label].end(), std::greater<float>());

		int i, j;
		i = j = 0;
		while (true)
		{
			if (TPScores[label].size() > 0 && FPScores[label].size() > 0)
			{
				if (TPScores[label][i] > FPScores[label][j])
				{
					tpVec.push_back(1);
					fpVec.push_back(0);
					i++;
				}
				else //(TPScores[label][i] < FPScores[label][j])
				{
					tpVec.push_back(0);
					fpVec.push_back(1);
					j++;
				}
			}
			if (i == TPScores[label].size() && j < FPScores[label].size()) //No more TPScore
			{
				while (j < FPScores[label].size())
				{
					tpVec.push_back(0);
					fpVec.push_back(1);
					j++;
				}
				break;
			}
			if (j == FPScores[label].size() && i < TPScores[label].size()) //No more TPScore
			{
				while (i < TPScores[label].size())
				{
					tpVec.push_back(1);
					fpVec.push_back(0);
					i++;
				}
				break;
			}
			if (FPScores[label].size() == 0 && TPScores[label].size() == 0)
				return 0.0;
		}

		int size = tpVec.size();

		int *tpCumsum = new int[size];
		std::partial_sum(tpVec.begin(), tpVec.end(), tpCumsum);

		int *fpCumsum = new int[size];
		std::partial_sum(fpVec.begin(), fpVec.end(), fpCumsum);

		float *precision = new float[size];
		for (int i = 0; i < size; i++)
			precision[i] = tpCumsum[i] / (float)(tpCumsum[i] + fpCumsum[i]);

		float *recall = new float[size];
		for (int i = 0; i < size; i++)
			recall[i] = tpCumsum[i] / (float)(numTp + numFn);

		float ap = 0.0;
		for (float t = 0.0; t <= 1; t += 0.1)
		{
			float max = -1.0;
			for (int i = 0; i < size; i++)
				if (recall[i] >= t)
					if (precision[i] >= max)
						max = precision[i];

			if (max == -1.0)
				max = 0.0;

			ap = ap + max / 11.0;
		}

		delete tpCumsum;
		delete fpCumsum;
		delete precision;
		delete recall;

		return ap;
	}

	std::string ActionRecognition::getFileName(std::string videoName, cv::FileStorage &params)
	{
		short i = 1;
		std::string ext;
		bool fileExists = false;
		std::string outputFold, path;
		// Open modelFile
		params["featureOutput"] >> outputFold;
		//system((mkdir + outputFold).c_str());

		if (saveBinaryFile)
			ext = ".bin";
		else
			ext = ".yml";

		path = outputFold + "\\" + videoName + ext;

		do
		{
			if (FILE *file = fopen(path.c_str(), "r"))
			{
				char buffer[3];
				fileExists = true;
				itoa(i++, buffer, 10);
				path = outputFold + "\\" + videoName + "_" + buffer + ext;
				fclose(file);
			}
			else
				fileExists = false;

		} while (fileExists);
		
		return path;
	}

	std::vector<std::string> splitString(std::string str, char delimiter)
	{
		std::vector<std::string> internal;
		std::stringstream ss(str);
		std::string tok;

		while (getline(ss, tok, delimiter))
			internal.push_back(tok);

		return internal;
	}

	std::vector<int> splitTemporalScales(std::string str, char delimiter)
	{
		std::vector<int> internal;
		std::stringstream ss(str);
		std::string tok;

		while (getline(ss, tok, delimiter)) {
			int ts = atoi(tok.c_str());
			internal.push_back(ts);
		}

		return internal;
	}
}