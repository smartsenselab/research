#include "dummy.hpp"
#include "descriptors\co_occurrence_general.hpp"
#include "descriptors/descriptor_temporal.hpp"
#include "core\cube.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iomanip>

int main() {
	
	int temporalSize = 5;
	ssig::Cube cuboid(0, 0, 2, 8, 8, 3);

	auto cuboidRoi = ssig::Cube(0, 0, 0, 16, 16, 8);
	auto intersection = cuboidRoi & cuboid;

	if (intersection != cuboid) {
		std::runtime_error(
			"Invalid cuboid, its intersection with the images are" +
			std::string("different than the cuboid itself"));
	}
	
	return 0;
}
