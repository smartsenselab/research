
#include "visual_dictionary.hpp"

//Carlos Caetano Research
namespace ccr
{

	std::vector<size_t> GenerateRandomPermutation(size_t n, size_t k);

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

	// function to return the dictionary
	void VisualDictionary::buildDictionary() {
		
		if (data.size() == 0)
		{
			std::cerr << "Set data first" << std::endl;
			exit(1);
		}

		std::vector<size_t> index;
		cv::Mat_<float> cluster(nCWs, data[0].cols);
		cv::Mat label;

		if (method == VisualDictionaryMethod::Random) {

			// Generate a random permutation of BigMatrix.
			index = GenerateRandomPermutation((size_t)data.size(), (size_t)nCWs);

			// Creates the dictionary
			for (int i = 0; i < nCWs; i++)
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

	// compute bag
	void VisualDictionary::computeBag(cv::Mat_<float> &data, cv::Mat_<float> &bag) {

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

	float VisualDictionary::distance(cv::Mat_<float> dict, cv::Mat_<float> data) {

		// Calculate distance
		float dist = 0;

		for (int i = 0; i < dict.cols; i++) {

			dist = dist + pow(dict.at<float>(0, i) - data.at<float>(0, i), 2);
		}
		return (dist / (dict.cols));
	}

	void VisualDictionary::addFeatureVectors(std::string &path) {
		int rows, cols;
		cv::FileStorage storageFeat;
		cv::FileNode node, n1;

		//Loading feature
		storageFeat.open(path, cv::FileStorage::READ);
		if (storageFeat.isOpened() == false)
			std::cerr << "Invalid file storage " << (path + "!") << std::endl;

		node = storageFeat.root();
		n1 = node["ActionRecognitionFeatures"];
		node = n1["Features"];
		node["rows"] >> rows;
		node["cols"] >> cols;
		storageFeat.release();

		for (int i = 0; i < rows; i++)
			this->data.push_back(FeatureIndex(path, rows, cols, i));
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

}