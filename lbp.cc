#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

#asd

REGISTER_OP("LBP")
    .Input("to_zero: int16")
    .Output("zeroed: int16")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
class LBPOp : public OpKernel {
 public:
  explicit LBPOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.matrix<int16>();

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->matrix<int16>();
	
	 int rows = input_tensor.shape().dim_size(0);
	 int columns = input_tensor.shape().dim_size(1);
	 
	 std::cout <<input_tensor.shape().dim_size(0)<< '\n';
	 std::cout << input.size()/rows<< '\n';
	 std::cout <<input.size()<< '\n';
	 
	 

	int my_array[rows][columns];
	int z = 0;
	for(int row = 0; row < rows; ++row)
    {
       for(int col = 0; col < columns; ++col)
       {
           my_array[row][col]=input(z);
		   z++;
       }
    }

	int lbp_array[rows][columns];
	
    unsigned center = 0;
    unsigned center_lbp = 0;
    for (int i = 1; i < rows-1; i++) {
    	for (int y = 1; y < columns-1; y++) {
    		center = my_array[i][y];
    		
    		center_lbp = 0; 
		    		
		if (center <= my_array[i-1][y-1])
		center_lbp += 1;
		
		if (center <= my_array[i-1][y])
		center_lbp += 2;

		if (center <= my_array[i-1][y+1])
		center_lbp += 4;

		if (center <= my_array[i][y-1])
		center_lbp += 8;

		if (center <= my_array[i][y+1])
		center_lbp += 16;

		if (center <= my_array[i+1][y-1])
		center_lbp += 32;

		if (center <= my_array[i+1][y])
		center_lbp += 64;

		if (center <= my_array[i+1][y+1])
		center_lbp += 128;
		
		 
    		lbp_array[i][y] = center_lbp;
    	}
    }
	
	int i=0;
	for(int row = 0; row < rows; ++row)
    {
       for(int col = 0; col < columns; ++col)
       {
           output_flat(i)=lbp_array[row][col];
		   i++;
       }
    }
	
	
  }
};

REGISTER_KERNEL_BUILDER(Name("LBP").Device(DEVICE_CPU), LBPOp);