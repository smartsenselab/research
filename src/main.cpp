#include "dummy.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "core\cube.hpp"
#include "action_recognition.hpp"

int main(int argc, char *argv[]) {
	
	////// TESTE //////
	//ccr::ActionRecognition *ar = new  ccr::ActionRecognition("arquivos//params.yml");
	//ar->execute();
	//delete ar;
	///////////////////

	if (argc > 1)
	{
		ccr::ActionRecognition *ar = new  ccr::ActionRecognition(argv[1]);
		ar->execute();
		delete ar;
	}
	else
		std::cout << argv[0] << ": Missing parameter file." <<  std::endl;

	return 0;
}
