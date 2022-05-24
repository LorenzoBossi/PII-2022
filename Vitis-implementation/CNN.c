#include "CNN.h"
#include "weights.h"
#include <stdio.h>
void pool(float input[CONVOLUTEDSIZE * CONVOLUTEDSIZE],
          float buffer[POOLEDSIZE * POOLEDSIZE]);
void init_tensor(float tensor[INPUTIMAGESIZE * INPUTIMAGESIZE],
                 float input[INPUTIMAGESIZE * INPUTIMAGESIZE]);
int Max(float probability[BIASSIZE]);
void Convolution(float tensor[INPUTIMAGESIZE * INPUTIMAGESIZE],
                 int kernelNumber,
                 float convoluted[CONVOLUTEDSIZE * CONVOLUTEDSIZE]);
void init_temp_array(float bias[BIASSIZE], float tempBias[BIASSIZE]);
void init_maxPooled(float buffer[POOLEDSIZE * POOLEDSIZE]);

void FullyConnected(float tensor[NUMBEROFKERNELS * TENSORSIZE],
                    int nKernel,
                    float result[BIASSIZE]);
void init_result(float result[BIASSIZE]);
void update_result(float result[BIASSIZE],
				   float fc[NUMBEROFKERNELS]);
void init(float result[BIASSIZE],
		float tensor[INPUTIMAGESIZE * INPUTIMAGESIZE],
		 float input[INPUTIMAGESIZE * INPUTIMAGESIZE]);
int CNN(float tensor[INPUTIMAGESIZE * INPUTIMAGESIZE]) {
  int i, j, f = 0, z;

  float finalLayer[FLATTENSIZE];

  float result[NUMBEROFKERNELS];
  float tempConvoluted[CONVOLUTEDSIZE * CONVOLUTEDSIZE];
  float tempMaxPooled[POOLEDSIZE * POOLEDSIZE];
  float fc[NUMBEROFKERNELS][BIASSIZE];
  float input[INPUTIMAGESIZE * INPUTIMAGESIZE];

  init(result,tensor,input);

  Convolution(input, 0, tempConvoluted);
  pool(tempConvoluted, tempMaxPooled);
  FullyConnected(tempMaxPooled, 0, fc[0]);

  SINGLE_KERNEL:
  for (i = 1,j=1; i < NUMBEROFKERNELS; j++,i++) {
#pragma HLS unroll
    Convolution(input, j, tempConvoluted);
    update_result(result,fc[i-1]);
    pool(tempConvoluted, tempMaxPooled);
    FullyConnected(tempMaxPooled, i, fc[i]);

  }
  update_result(result,fc[9]);

  float sum;
  int max;
  float tempsum1[NUMBEROFKERNELS / 2];
  float tempsum2[NUMBEROFKERNELS / 4];
  float tempsum;
  float singleElement[BIASSIZE];

SOFTMAX_LOADING_FOR:
  for (i = 0; i < BIASSIZE; i++) {
    singleElement[i] = exp(result[i]);
  }

SOFTMAX_SUM_FOR_1:
  for (j = 0, i = 0; i < BIASSIZE / 2; i++, j = j + 2) {
    tempsum1[i] = singleElement[j] + singleElement[j + 1];
  }

SOFTMAX_SUM_FOR_2:
  for (j = 0, i = 0; i < BIASSIZE / 4; i++, j = j + 2) {
    tempsum2[i] = tempsum1[j] + tempsum1[j + 1];
  }

  tempsum = tempsum2[0] + tempsum2[1];
  sum = tempsum + tempsum1[4];

SOFTMAX_FOR:
  for (i = 0; i < BIASSIZE; i++) {
    result[i] = 100 * singleElement[i] / sum;
  }

  return Max(result);
}


void init(float result[BIASSIZE],
		float tensor[INPUTIMAGESIZE * INPUTIMAGESIZE],
		 float input[INPUTIMAGESIZE * INPUTIMAGESIZE]){
#pragma HLS dataflow
	init_result(result);
	init_tensor(tensor,input);
}


void init_result(float result[BIASSIZE]) {
	int i;
	INIT_BIAS:for (i = 0; i < NUMBEROFKERNELS; i++) {
	    result[i] = denseBias[i];
	  }
}


void update_result(float result[BIASSIZE],float fc[NUMBEROFKERNELS]){
	int i,z;
	SUM1:
	  for(i=0;i<NUMBEROFKERNELS;i++){
	      result[i] += fc[i];
	    }

}


int Max(float probability[10]) {

	int maxi;
	float max;
	int i;
	max=0;
	for(i=0;i<10;i++){
#pragma HLS pipeline II=2
		if(probability[i]> max){
			max=probability[i];
			maxi=i;
		}
	}
	return maxi;
}


void init_tensor(float tensor[INPUTIMAGESIZE * INPUTIMAGESIZE],
                 float input[INPUTIMAGESIZE * INPUTIMAGESIZE]) {
  int i;

INIT_TENSOR:
  for (i = 0; i < INPUTIMAGESIZE * INPUTIMAGESIZE;
       i++)  // for each row of the convoluted matrix
  {
#pragma HLS unroll
    input[i] = tensor[i];
  }
}


void init_maxPooled(float buffer[POOLEDSIZE * POOLEDSIZE]) {
  int i, j;
INIT_MAX_POOLED:
  for (j = 0; j < POOLEDSIZE * POOLEDSIZE; j++) {
    buffer[j] = 0;
  }
}


void Convolution(float tensor[INPUTIMAGESIZE * INPUTIMAGESIZE],
                 int kernelNumber,
                 float convoluted[CONVOLUTEDSIZE * CONVOLUTEDSIZE]) {
  float sum;
  int ZI_INPUTSIZE, Z_KERNELSIZE;
  int i, f, z, j;
  float stemp;
  float tempsum[KERNELSIZE * KERNELSIZE];

MATRIX_ROW:
  for (i = 0; i < CONVOLUTEDSIZE;
       i++) {  // for each row of the convoluted matrix
#pragma HLS pipeline II=98

  MATRIX_COLUMN:
    for (f = 0; f < CONVOLUTEDSIZE;
         f++) {
#pragma HLS unroll
    	// for each column of the convoluted matrix
      sum = kernelsBias[kernelNumber];
    KERNEL_COLUMN:
      for (z = 0; z < KERNELSIZE; z++)  // for each row of the kernel
      {

        ZI_INPUTSIZE = (z + i) * INPUTIMAGESIZE;
        Z_KERNELSIZE = z * KERNELSIZE;
      KERNEL_ROW:
        for (j = 0; j < KERNELSIZE; j++)  // for each column of the kernel
        {
          sum += (tensor[ZI_INPUTSIZE + j + f] *
                  kernels[kernelNumber][Z_KERNELSIZE + j]);
        }
      }

      if (sum < 0.0)
        convoluted[i * CONVOLUTEDSIZE + f] = 0;
      else
        convoluted[i * CONVOLUTEDSIZE + f] = sum;
    }
  }
}


void pool(float input[CONVOLUTEDSIZE * CONVOLUTEDSIZE],
          float buffer[POOLEDSIZE * POOLEDSIZE]) {
  int i, f, z, j, buffer_index, input_index;
  float temp_buffer, temp_input;

  init_maxPooled(buffer);

POOLED_ROW:
  for (i = 0; i < POOLEDSIZE; i++)  // for each row of the convoluted matrix
  {
#pragma HLS pipeline II=35
  POOLED_COLUMN:
    for (f = 0; f < POOLEDSIZE;
         f++)  // for each column of the convoluted matrix
    {
#pragma HLS unroll

    STRIFE_ROW:
      for (z = 0; z < STRIFESIZE; z++)  // for each row of the kernel
      {
      STRIFE_COLUMN:
        for (j = 0; j < STRIFESIZE; j++)  // for each column of the kernel
        {
          buffer_index = i * POOLEDSIZE + f;
          temp_buffer = buffer[buffer_index];

          input_index =
              (z + i * STRIFESIZE) * CONVOLUTEDSIZE + j + f * STRIFESIZE;
          temp_input = input[input_index];

          if (temp_buffer <
              temp_input)  // check if the value is higher then max
          {
            buffer[buffer_index] = temp_input;
          }
        }
      }
    }
  }
}



void init_buffer(float buffer[POOLEDSIZE * POOLEDSIZE]) {
  int i;

INIT_BUFFER:
  for (i = 0; i < POOLEDSIZE * POOLEDSIZE;
       i++)  // for each row of the convoluted matrix
  {
    buffer[i] = 0;
  }
}


void init_input(float tensor[CONVOLUTEDSIZE * CONVOLUTEDSIZE],
                float input[CONVOLUTEDSIZE * CONVOLUTEDSIZE]) {
  int j;
INIT_INPUT:
  for (j = 0; j < CONVOLUTEDSIZE * CONVOLUTEDSIZE; j++)
    input[j] = tensor[j];
}


void FullyConnected(float tensor[TENSORSIZE],
                    int nKernel,
                    float result[BIASSIZE]) {
  int i, j, f,y, weightIndex, weightBiasIndex;
  float tempFlatten;
  float tempBiasValue;
  float tempWeight;
  f = 0;
  for (i = 0; i < 10; i++) {
    result[i] = 0;
  }

  
FULLY_CONNECTED_SINGLE_ELEMENT:
  for (j = 0; j < TENSORSIZE; j++) {
#pragma HLS pipeline II=10
    tempFlatten = tensor[j];
    if (tensor[j] != 0) {
      weightBiasIndex = j * BIASSIZE;
    FULLY_CONNECTED_SINGLE_ELEMENT_WEIGHTS:
      for (i = weightBiasIndex,y=0, f = 0; y < BIASSIZE;
           i++,y++)  // error with index
      {
        tempWeight = dense[nKernel][i];
        result[f] = result[f] + (tempFlatten * tempWeight);
        f++;
      }
    }
  }
}


void init_temp_array(float bias[BIASSIZE], float tempBias[BIASSIZE]) {
  int j;
INIT_INPUT:
  for (j = 0; j < BIASSIZE; j++)
    tempBias[j] = bias[j];
}
