#ifndef MESH_MANIPULATOR_H_
#define MESH_MANIPULATOR_H_

#include <stdbool.h>
#include <string>
#include <vtkSmartPointer.h>
#include <vtkOBJImporterInternals.h>
#include <vtkTransformPolyDataFilter.h>
#include <itkLandmarkBasedTransformInitializer.h>

using Element = double;
const size_t Dimension = 3;
using Point = itk::Point<Element, Dimension>;
using Image = itk::Image<Element, Dimension>;
using Transform = itk::AffineTransform<Element, Dimension>;
using ITKPointType = itk::Point<Element, Dimension>;

class Manipulator 
{
	public:
		Manipulator();
		Manipulator(std::string &file);
		bool ManipulateObj();
		void VisualizeResults(vtkSmartPointer<vtkOBJImporter> reader);

		// Getters and setters
		void SetObjFilePath(std::string &file);
		std::string GetObjFilePath();

	private:
		bool ReadObj(vtkSmartPointer<vtkOBJPolyDataProcessor> reader);
		vtkSmartPointer<vtkTransformPolyDataFilter> AlignObj(vtkSmartPointer<vtkPolyData> reader, vtkSmartPointer<vtkPoints> moving_points, vtkSmartPointer<vtkPoints> fixed_points);
		void WriteObj(vtkSmartPointer<vtkTransformPolyDataFilter> mesh);

		// Private data members
		std::string file_path;
};
#endif //MESH_MANIPULATOR_H_