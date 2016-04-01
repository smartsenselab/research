
#include "visual_dictionary.hpp"

//Carlos Caetano Research
namespace ccr
{

	std::vector<size_t> GenerateRandomPermutation(size_t n, size_t k);
	std::vector<int> GenerateRandomPermutation2(size_t n, size_t k);
	cv::Mat matRead(const std::string& filename, std::string &label);

	VisualDictionary::VisualDictionary() {
		nCWs = 10;
		method = VisualDictionaryMethod::Random;
	}

	VisualDictionary::VisualDictionary(int nCWs, VisualDictionaryMethod method) {
		this->nCWs = nCWs;
		this->method = method;

		beforeProcess();
	}

	VisualDictionary::~VisualDictionary() {

	}


	void VisualDictionary::beforeProcess() {

		if (nCWs <= 0)
			std::cerr << "Number of codewords must be positive!" << std::endl;
		if (method != VisualDictionaryMethod::Random && method != VisualDictionaryMethod::Kmeans)
			std::cerr << "Parameter 'method' has to be set Random or Kmeans" << std::endl;
	}

	void VisualDictionary::copyCodeWord(int i, cv::Mat_<float> &cluster, std::vector<size_t> &index)
	{
		cv::Mat feature;
		cv::FileStorage storageFeat;
		cv::FileNode node, n1;

		//Loading feature
		storageFeat.open(data[index[i]].path, cv::FileStorage::READ);
		if (storageFeat.isOpened() == false)
			std::cerr << "Invalid file storage " << (data[index[i]].path + "!") << std::endl;

		node = storageFeat.root();
		n1 = node["ActionRecognitionFeatures"];
		n1["Features"] >> feature;
		storageFeat.release();

		feature.row(data[index[i]].index).copyTo(cluster.row(i));
	}

	void VisualDictionary::copyCodeWord2(int i, int pos, cv::Mat_<float> &cluster, std::vector<int> &index)
	{
		cv::Mat feature;
		cv::FileStorage storageFeat;
		cv::FileNode node, n1;
		int randInt;
		std::string label;

		// binary file
		if (pathFiles[i][pathFiles[i].size() - 1] == 'n')
		{
			feature = matRead(pathFiles[i], label);
		}
		else
		{
			//Loading feature
			storageFeat.open(pathFiles[i], cv::FileStorage::READ);
			if (storageFeat.isOpened() == false)
				std::cerr << "Invalid file storage " << (pathFiles[i] + "!") << std::endl;

			node = storageFeat.root();
			n1 = node["ActionRecognitionFeatures"];
			n1["Label"] >> label;
			n1["Features"] >> feature;
			storageFeat.release();
		}

		FeatureIndex fi((pathFiles[i]), label, feature.rows, feature.cols, 0); // 0 is  a dummy number
		featuresProperties.push_back(fi);

		//////////////////// This is for random seed //////////////////////
		//std::random_device rd;     // only used once to initialise (seed) engine
		//std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
		//std::uniform_int_distribution<int> distribution(0, feature.rows); // min, max. Guaranteed unbiased

		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(0, feature.rows - 1);

		for (int j = 0; j < index[i]; j++)
		{
				randInt = distribution(generator);
				feature.row(randInt).copyTo(cluster.row(pos++));
		}

	}

	// function to return the dictionary
	void VisualDictionary::buildDictionary() {

		if (data.size() == 0)
		{
			std::cerr << "Set data first" << std::endl;
			exit(1);
		}

		std::vector<size_t> index;
		std::vector<std::thread> threads;
		cv::Mat_<float> cluster(nCWs, data[0].cols);

		if (method == VisualDictionaryMethod::Random) {

			index = GenerateRandomPermutation((size_t)data.size(), (size_t)nCWs);

			// Creates the dictionary
			int i = 0;
			int percent;
			std::cout << " ";
			while (i < nCWs)
			{
				percent = (i * 100) / nCWs;
				std::cout << percent << "%";
				std::vector<int> codeWordVector;
				for (int cores = 0; cores < std::thread::hardware_concurrency() && i < nCWs; cores++, i++)
					codeWordVector.push_back(i);

				for (auto& p : codeWordVector)
					threads.push_back(std::thread(&VisualDictionary::copyCodeWord, this, p, cluster, index));

				for (auto& th : threads)
					th.join();

				threads.clear();

				if (percent > 9)
					std::cout << "\b\b\b";
				else
					std::cout << "\b\b";
			}
			std::cout << "100%";
			std::cout << "\b\b\b\b\b";
		}
		else if (method == VisualDictionaryMethod::Kmeans) {
			// Define criteria
			cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 0.001);

			// Apply kmeans()
			//cv::kmeans(data, nCWs, label, criteria, 1, cv::KMEANS_RANDOM_CENTERS, cluster);
			// Está comentado pois teria que ler todas as features do disco e colcoar em uma Matriz data... não tem memória suficiente para subir todas
		}

		dictionary = cluster.clone();
		data.clear(); //data is no more needed
	}

	void VisualDictionary::buildDictionary2(std:: string path) {

		DIR *dir = 0;
		std::string file;
		std::vector<int> index;
		struct dirent *featFile = 0;
		std::vector<std::thread> threads;
		cv::FileStorage storageFeat;
		cv::FileNode node, n1;
		int cols;

		dir = opendir(path.c_str());

		if (dir == 0) {
			std::cerr << "Impossible to open directory." << std::endl;
			exit(1);
		}
		
		pathFiles.clear();
		while (featFile = readdir(dir))
			if (featFile->d_type == isFile)
				pathFiles.push_back(path + "\\" + featFile->d_name);
		
		closedir(dir);
		index = GenerateRandomPermutation2((size_t)pathFiles.size(), (size_t)nCWs);		

		// binary file
		if (pathFiles[0][pathFiles[0].size() - 1] == 'n')
		{
			std::string dummy;
			cv::Mat feature = matRead(pathFiles[0], dummy);
			cols = feature.cols;
		}
		else
		{
			storageFeat.open(pathFiles[0], cv::FileStorage::READ);
			if (storageFeat.isOpened() == false)
				std::cerr << "Invalid file storage " << (pathFiles[0] + "!") << std::endl;

			node = storageFeat.root();
			n1 = node["ActionRecognitionFeatures"];
			node = n1["Features"];
			node["cols"] >> cols;
			storageFeat.release();
		}

		cv::Mat_<float> cluster(nCWs, cols);
		if (method == VisualDictionaryMethod::Random) {
			// Creates the dictionary
			int i = 0;
			int percent;
			int pos = 0;
			std::cout << " ";
			while (i < pathFiles.size())
			{
				percent = (i * 100) / pathFiles.size();
				std::cout << percent << "%";
				std::vector<std::pair<int, int>> codeWordVector;
				//for (int cores = 0; cores < std::thread::hardware_concurrency() && i < pathFiles.size(); cores++, i++)
				int cores = 0;
				while (cores < std::thread::hardware_concurrency() && i < pathFiles.size())
				{
					if (index[i] == 0)
					{
						i++;
						continue;
					}
					
					std::pair<int, int> p(i, pos);
					codeWordVector.push_back(p);
					pos += index[i++];
					cores++;
				}

				for (auto& p : codeWordVector)
					threads.push_back(std::thread(&VisualDictionary::copyCodeWord2, this, p.first, p.second, cluster, index));

				for (auto& th : threads)
					th.join();

				threads.clear();

				if (percent > 9)
					std::cout << "\b\b\b";
				else
					std::cout << "\b\b";
			}
			std::cout << "100%";
			std::cout << "\b\b\b\b\b";
		}
		else if (method == VisualDictionaryMethod::Kmeans) {
			// Define criteria
			cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 0.001);

			// Apply kmeans()
			//cv::kmeans(data, nCWs, label, criteria, 1, cv::KMEANS_RANDOM_CENTERS, cluster);
			// Está comentado pois teria que ler todas as features do disco e colcoar em uma Matriz data... não tem memória suficiente para subir todas
		}

		dictionary = cluster.clone();
		data.clear(); //data is no more needed
		pathFiles.clear();
	}

	// compute bag
	void VisualDictionary::computeBag(cv::Mat &data, cv::Mat &bag) {

		float dist = FLT_MAX, aux, idx;

		if (data.cols != dictionary.cols) {
			std::cerr << "Data size and dictionary size doesn't match!" << std::endl;
		}

		for (int i = 0; i < dictionary.rows; i++) {

			aux = distance(dictionary.row(i), data);

			if (aux <= dist) {
				dist = aux;
				idx = static_cast<float>(i);
			}
		}
		bag.push_back(dist);
		bag.push_back(idx);
	}

	float VisualDictionary::distance(cv::Mat dict, cv::Mat data) {

		// Calculate distance
		float dist = 0;

		for (int i = 0; i < dict.cols; i++) {

			dist = dist + pow(dict.at<float>(0, i) - data.at<float>(0, i), 2);
		}
		return (dist / (dict.cols));
	}

	void VisualDictionary::addFeatureVectors(FeatureIndex &p) {

		for (int i = 0; i < p.rows; i++)
			this->data.push_back(FeatureIndex(p.path, p.label, p.rows, p.cols, i));
	}

	// Essa função deve mudar
	void VisualDictionary::addFeatureVectors(std::vector<FeatureIndex> fi, std::vector<std::string> &labels) {
		int i;

		// check whether the number of feature vectors is equal to the number of labels
		if (fi.size() != labels.size())
			std::cerr << "Inconsistant number of labels and feature vectors (must be equal)" << std::endl;

		for (i = 0; i < fi.size(); i++)
		{
			this->data.push_back(fi[i]);
			this->labels.push_back(labels[i]);
		}

	}

	// save dictionary
	void VisualDictionary::save(cv::FileStorage &storage) {

		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;

		storage << "Dictionary" << "{";
		storage << "nCWs" << nCWs;
		storage << "dict" << dictionary;
		storage << "}";
	}

	// load dictionary
	void VisualDictionary::load(const cv::FileNode &node, cv::FileStorage &storage) {

		cv::FileNode n;

		n = node["Dictionary"];
		n["nCWs"] >> nCWs;
		n["dict"] >> dictionary;
	}

	int VisualDictionary::getnCWs() { return nCWs; }
	std::vector<FeatureIndex> VisualDictionary::getFeaturesProperties() { return featuresProperties; }
	void VisualDictionary::clearFeaturesProperties() { featuresProperties.clear(); }

	std::vector<size_t> GenerateRandomPermutation(size_t n, size_t k) {
		std::vector<size_t> myvector;
		std::vector<size_t> newvector;
		std::vector<size_t>::iterator it;
		int i;

		// set some values
		for (i = 0; i < n; i++) {
			myvector.push_back(i);
		}

		// using built-in random generator:
		random_shuffle(myvector.begin(), myvector.end());

		for (i = 0; i < k; i++)
			newvector.push_back(myvector.at(i));

		return newvector;
	}

	std::vector<int> GenerateRandomPermutation2(size_t n, size_t k) {
		
		int randInt;
		std::vector<int> myvector(n, 0);
		
		//////////////////// This is for random seed //////////////////////
		//std::random_device rd;     // only used once to initialise (seed) engine
		//std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
		//std::uniform_int_distribution<int> distribution(0, feature.rows); // min, max. Guaranteed unbiased
		
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(0, n - 1);

		for (int i = 0; i < k; i++)
		{
			randInt = distribution(generator);
			myvector[randInt]++;
		}

		return myvector;
	}


	cv::Mat matRead(const std::string& filename, std::string &label)
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
		delete [] temp;

		fs.read((char*)&rows, sizeof(int));         // rows
		fs.read((char*)&cols, sizeof(int));         // cols
		fs.read((char*)&type, sizeof(int));         // type
		fs.read((char*)&channels, sizeof(int));     // channels

		// Data
		cv::Mat mat(rows, cols, type);
		fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

		return mat;
	}

}