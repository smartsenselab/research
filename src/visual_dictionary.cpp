
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
			std::cerr << "Number of codewords must be positive!";
		if (method != VisualDictionaryMethod::Random && method != VisualDictionaryMethod::Kmeans)
			std::cerr << "Parameter 'method' has to be set Random or Kmeans";
	}

	// function to return the dictionary
	void VisualDictionary::buildDictionary() {

		std::vector<size_t> index;
		cv::Mat_<float> cluster(nCWs, data.cols);
		cv::Mat label;

		if (data.rows == 0)
			std::cerr << "Set data first";

		if (method == VisualDictionaryMethod::Random) {

			// Generate a random permutation of BigMatrix.
			index = GenerateRandomPermutation((size_t)data.rows, (size_t)nCWs);

			// Creates the dictionary
			for (int i = 0; i < nCWs; i++)
			{
				data.row((int)index[i]).copyTo(cluster.row(i));
			}
		}
		else if (method == VisualDictionaryMethod::Kmeans) {
			// Define criteria
			cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 0.001);

			// Apply kmeans()
			cv::kmeans(data, nCWs, label, criteria, 1, cv::KMEANS_RANDOM_CENTERS, cluster);
		}

		dictionary = cluster.clone();
		data.release(); //data is no more needed
	}

	// compute bag
	void VisualDictionary::computeBag(cv::Mat_<float> &data, cv::Mat_<float> &bag) {

		float dist = FLT_MAX, aux, idx;

		if (data.cols != dictionary.cols) {
			std::cerr << "Data size and dictionary size doesn't match!\n";
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

	void VisualDictionary::addFeatureVectors(const cv::Mat_<float> &data) {

		this->data.push_back(data);
	}

	void VisualDictionary::addFeatureVectors(const cv::Mat_<float> &data, const std::vector<std::string> &labels) {
		int i;

		// check whether the number of feature vectors is equal to the number of labels
		if (data.rows != (int)labels.size())
			std::cerr << "Inconsistant number of labels and feature vectors (must be equal)";

		this->data.push_back(data);

		for (i = 0; i < data.rows; i++)
			this->labels.push_back(labels[i]);
	}

	// save dictionary
	void VisualDictionary::save(cv::FileStorage &storage) {

		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!";

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