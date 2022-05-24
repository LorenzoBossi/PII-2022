
#include<math.h>
#include<stdio.h>
#include"weights.h"

#define STRIFESIZE 3      // How many spaces do we have to move while max pooling
#define KERNELSIZE 7      // dimension of the square KernelMatrix
#define NUMBEROFKERNEL 10
#define INPUTIMAGESIZE 28

#define expectedConvolutionSize 22 //the dimendsion of the matrix after convolution
#define expectedMaxPolledSize 7 //dimension of the matrix after max pooling (formula ceil(expected/strife))
#define MaxPooledElements  expectedMaxPolledSize*expectedMaxPolledSize

#define FLbiasSize 10 //number of node of the final layer
#define FLweightsSize FLbiasSize*expectedMaxPolledSize*expectedMaxPolledSize*NUMBEROFKERNEL
#define FlattenSize expectedMaxPolledSize*expectedMaxPolledSize*NUMBEROFKERNEL

#define LEN 13
#define NUMBEROFPIXELS 784 //28*28

#define im_filename "timages_mnist.idx3-ubyte"
#define label_filename "tlabels_mnist.idx1-ubyte"

void Convolution(float* tensor, int size, float* kernel,int kernelSize, float bias, float* convoluted, int expectedSize);
void MaxPooling(float* tensor, int size, int strifesize, float* maxPooled, int expectedSize);
void FullyConnected(float** tensor, int tensorSize,int numberOfTensor, float* weight, int weightSize, float* bias, int biasSize, float* layer);
int open_files( char * lab_filename,FILE ** ifp,FILE ** lfp);
int get_number(FILE ** ifp,FILE ** lfp,float * img, int * label);
float relu(float value);
void softmax(float * tensor, int tensorsize);


int main(int argc,char * argv[])
{
	int inputnumber;
	int i,j;
	int label;
    float inputImage[INPUTIMAGESIZE*INPUTIMAGESIZE];
	FILE * imagesfp;
	FILE * labelsfp;
	printf("Inserisci il numero dell' immagine (1 - 10000) :\n");
	scanf("%d",&inputnumber);

    //load an image
	if(!open_files( label_filename,&imagesfp,&labelsfp))
	{
		printf("unable to open file\n");
		return 0;
	}
	for(i=0;i<inputnumber;i++)
		get_number(&imagesfp,&labelsfp,inputImage, &label);

    //print the image and the label
	for(i=0;i<INPUTIMAGESIZE;i++)
	{
		for(j=0;j<INPUTIMAGESIZE;j++)
		{
			printf("%f ",inputImage[i*INPUTIMAGESIZE+j]);
		}
		printf("\n");
	}
	printf("\n");
    printf("\nlabel: %d\n\n",label);


	float * ccLayer[NUMBEROFKERNEL];
	float pooled[NUMBEROFKERNEL][expectedMaxPolledSize*expectedMaxPolledSize];
	float tempConvolution[expectedConvolutionSize*expectedConvolutionSize];
	// apply the convolution and max pooling for each kernel
	for (i = 0; i < NUMBEROFKERNEL; i++)                                
	{
		ccLayer[i]=&pooled[i][0];
		Convolution(inputImage, INPUTIMAGESIZE, kernels[i],KERNELSIZE,kernelsBias[i], tempConvolution, expectedConvolutionSize);
		MaxPooling(tempConvolution, expectedConvolutionSize, STRIFESIZE, ccLayer[i], expectedMaxPolledSize);
	}
	// Fully connected layer
	float FLayer[FLbiasSize]; 
	FullyConnected(ccLayer, expectedMaxPolledSize * expectedMaxPolledSize, NUMBEROFKERNEL,dense,FLweightsSize,  denseBias, FLbiasSize, FLayer);


    //print results
	for(int i = 0; i < FLbiasSize; i++) {
		printf("Numero: %d  ,probabilità %f\n",i,FLayer[i]);
	}

	return 0;
}


// to save memory and time we treat the tensor as an array
// to access the array in the [x][y] position we use [x*size+y]
void Convolution(float* tensor, int size, float* kernel, int kernelSize, float bias, float* convoluted, int expectedSize)
{
    float sum;
    for (int i = 0; i < expectedSize; i++)// for each row of the convoluted matrix
    { 
        for (int f = 0; f < expectedSize; f++)// for each column of the convoluted matrix
        { 
            sum = 0;
            for (int z = 0; z < kernelSize && z + i < size; z++) // for each row of the kernel
            {
                for (int j = 0; j < kernelSize && j + f < size; j++) // for each column of the kernel
                {
					// multiply the tensor elelment with the right kernel element and adds it to sum;
                    sum = sum + (tensor[(z + i) * size + j + f] * kernel[z * kernelSize + j]); 
				}
			}
			convoluted[i * expectedSize + f] = relu(sum + bias);
        }
    }
}

void MaxPooling(float* tensor, int size, int strifesize, float* maxPooled, int expectedSize)
{
	float max;
    for (int i = 0; i < size; i=i+strifesize)// for each row of the convoluted matrix
    {
        for (int f = 0; f < size; f=f+strifesize)// for each column of the convoluted matrix
        { 
			max=0;
            for (int z = 0; z < strifesize && z + i < size; z++) // for each row of the kernel
            {
                for (int j = 0; j < strifesize && j + f < size; j++) // for each column of the kernel
                {
                    if (max < tensor[(z + i) * size + j + f])
                        max = tensor[(z + i) * size + j + f]; // check if the value is higer then max
                }
            }

            maxPooled[(i/strifesize) * expectedSize + f/strifesize] = max;
        }
    }
}

void FullyConnected(float** tensor, int tensorSize, int numberOfTensor, float* weight, int weightSize, float * bias, int biasSize, float* layer)
{
	int i,j,f;
	f=0;
	float flatten[FlattenSize];

		for(j=0;j<tensorSize;j++)
		{
			for(i=0;i<biasSize;i++)
			{
			flatten[f]=tensor[i][j];
			f++;
			}
		}

	for(j=0;j<FlattenSize;j++)
	{
		for(i=0;i<biasSize;i++)
		{
			layer[i] = layer[i] + (flatten[j] * weight[i + j*10]);
		}
	}

    for (i = 0; i < biasSize; i++)
	{
			layer[i] = layer[i] + bias[i];
	}
	softmax(layer,biasSize);

}


float relu(float value) {
		if(value<0.0)
			return 0.0;
		return value;
}

void sigmoid(float* tensor, int tensorsize){
	for (int i = 0; i < tensorsize; i++) {
		printf("tensor-> %f sigmoid %f",tensor[i],1/(1+exp(-tensor[i])));
		tensor[i]=1/(1+exp(-tensor[i]));
		printf("%d -> %f\n",i,tensor[i]);
	}
}
void softmax(float* tensor, int tensorsize) {
	float sum = 0;
	for (int i = 0; i < tensorsize; i++) {
		sum = sum +exp(tensor[i]);
	}
	for (int i = 0; i < tensorsize; i++) {
		tensor[i]=100*exp(tensor[i])/sum;
	}
}

int open_files( char * lab_filename,FILE ** ifp,FILE ** lfp)
{
	int par[4];
	//opening images file
	if( (*ifp=fopen(im_filename,"rb")) )
	{
		fread(par,sizeof(int),4,*ifp);
	}
	else{
		printf("unable to open file : %s\n",im_filename);
		return 0;
	}
	//opening label file
	if( (*lfp=fopen(lab_filename,"rb")) )
	{
		fread(par,sizeof(int),2,*lfp);
	}
	else{
		printf("unable to open file : %s\n",lab_filename);
		return 0;
	}

	return 1;
}


int get_number(FILE ** ifp,FILE ** lfp,float * img, int * label)
{

	int i,j;
	unsigned char buffer[INPUTIMAGESIZE][INPUTIMAGESIZE];
	fread((int *)buffer,sizeof(char),NUMBEROFPIXELS,*ifp);
	fread(label,sizeof(char),1,*lfp);
	if(feof(*ifp)||feof(*lfp))
		return 0;
	for(i=0;i<INPUTIMAGESIZE;i++)
	{
		for(j=0;j<INPUTIMAGESIZE;j++)
		{
			img[i*INPUTIMAGESIZE+j%INPUTIMAGESIZE]=buffer[i][j]/255.0;
		}
	}
	return 1;
}
