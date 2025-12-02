#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <glm/glm.hpp>
#include <iostream>
#include <fstream>
using namespace std;
using namespace glm;
unsigned char* Canny(unsigned char* buffer, int width, int height, int comps, float low, float high)
{
    unsigned char* denoised = new unsigned char[width*height*comps];
	memcpy(denoised,buffer, width*height*comps);
    //noise reduction
    for(int i=1;i<height-1;i++)
    {
      	for (int j=1;j<width-1;j++)
      	{
			
			int newValue = (int) ((1.0 * buffer[((i-1)*width+j-1)*comps] + 
								2.0 * buffer[((i-1)*width+j)*comps] +
								1.0 * buffer[((i-1)*width+j+1)*comps] +
								2.0 * buffer[((i)*width+j-1)*comps] + 
								4.0 * buffer[((i)*width+j)*comps] + 
								2.0 * buffer[((i)*width+j+1)*comps] + 
								1.0 * buffer[((i+1)*width+j-1)*comps] + 
								2.0 * buffer[((i+1)*width+j)*comps] + 
								1.0* buffer[((i+1)*width+j+1)*comps] )/16.0);
			newValue = clamp(newValue,0,255);
			denoised[((i)*width+j)*comps] = newValue; //r
			denoised[((i)*width+j)*comps+1] = newValue; //g
			denoised[((i)*width+j)*comps+2] = newValue; //b
		    denoised[((i)*width+j)*comps+3] = 255; //a
      	}
    }
    stbi_write_png("./pictures/Denoised.png", width, height, comps, denoised, width * comps);
	
	//sobel

    int gxMatrix[3][3] =
        {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
    int gyMatrix[3][3] =
        {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
        };
    //float should be enough
    float* magnitudes = (float*)malloc(width*height*sizeof(float));
    fill(magnitudes, magnitudes+width*height, 0.0f);
    float* directions = (float*)malloc(width*height*sizeof(float));
    fill(directions, directions+width*height, 0.0f);
    for(int i=1;i<height-1;i++)
    {
      	for (int j=1;j<width-1;j++)
      	{
            //correlation, not convolution
			float gxVal = 0.0f;
			float gyVal = 0.0f;
            for(int x=0;x<3;x++)
            {
                for(int y=0;y<3;y++)
                {
                    gxVal += gxMatrix[x][y] * (float) (denoised[((i+y-1)*width+j+x-1)*comps]) ;
                    gyVal += gyMatrix[x][y] * (float) (denoised[((i+y-1)*width+j+x-1)*comps]) ;
                }
            }
            //using built in glm functions
            vec2 gVec (gxVal, gyVal);
            magnitudes[i*width+j] = length(gVec);
            //round to nearest 45 degrees
            directions[i*width+j] = degrees(atan2(gyVal,gxVal));
      	}
    }
    //debug print
    FILE* directionText= fopen("Directions.txt", "w");
    for (int i=0;i<height;i++)
    {
        for (int j = 0; j < width ; j++)
        {
            //string s = to_string(floyd[(i*width +j)*comps] /16) ;
            fprintf(directionText,"%3.0f", directions[(i*width +j)]);
            fputc(',', directionText);
        }
        fputc('\n', directionText);
    }
    fclose(directionText);
    FILE* gradText= fopen("Gradient.txt", "w");
    for (int i=0;i<height;i++)
    {
        for (int j = 0; j < width ; j++)
        {
            //string s = to_string(floyd[(i*width +j)*comps] /16) ;
            fprintf(gradText,"%3.1f", magnitudes[(i*width +j)]);
            fputc(',', gradText);
        }
        fputc('\n', gradText);
    }
    fclose(gradText);

    unsigned char* grad =(unsigned char*) malloc(width*height*comps);
    
    for(int i=0;i<height;i++)
    {
      	for (int j=0;j<width;j++)
      	{
            int newVal = clamp((int)magnitudes[i*width+j],0,255);
            grad[(width*i+j)*comps]= newVal;
            grad[(width*i+j)*comps+1]= newVal;
            grad[(width*i+j)*comps+2]= newVal;
            grad[(width*i+j)*comps+3]= 255;
      	}
    }
    stbi_write_png("./pictures/Gradient.png", width, height,comps, grad, width * comps);
	
    //unsigned char*  nonmax = new unsigned char[width*height*comps];
    //non-max suppression
    for(int i=1;i<height-1;i++)
    {
      	for (int j=1;j<width-1;j++)
      	{
            //correlation, not convolution
			float direction = directions[i*width+j];
            if (direction<0) direction+=180;
            int degree = (int)round(direction/45.0)*45;
            float n1, n2;
            
            if(degree==0 || degree==180)
            {
                //top and bottom
                n1 = magnitudes[(i-1)*width+j];
                n2 = magnitudes[(i+1)*width+j];
            }
            else if(degree==45)
            {
                //rotate 45
                n1 = magnitudes[(i-1)*width+j-1];
                n2 = magnitudes[(i+1)*width+j+1];
            }
            else if(degree==90)
            {
                //left and right
                n1 = magnitudes[(i)*width+j-1];
                n2 = magnitudes[(i)*width+j+1];
            }
            else if(degree == 135)
            {
                //rotate another 45
                n1 = magnitudes[(i-1)*width+j+1];
                n2 = magnitudes[(i+1)*width+j-1];
            }
            //if it's maximum of 3, keep it; otherwise, set to 0
            int newVal = 0;
            float marginOfError = 0.0f;
            if((magnitudes[i*width+j] +marginOfError)>=n1 && (magnitudes[i*width+j] + marginOfError)>=n2)
            {
                newVal = clamp((int)magnitudes[i*width+j],0,255);
            }
            grad[(width*i+j)*comps]= newVal;
            grad[(width*i+j)*comps+1]= newVal;
            grad[(width*i+j)*comps+2]= newVal;
            grad[(width*i+j)*comps+3]= 255;
      	}
    }
    stbi_write_png("./pictures/NonMax.png", width, height, comps, grad, width * comps);
	

    //double treshlolding

    
    for(int i=0;i<height;i++)
    {
      	for (int j=0;j<width;j++)
      	{
            float ratio = grad[(width*i+j)*comps]/256.0;
            int newVal;
            if(ratio>high)
            {
                //strong edge
                newVal=255;
            }
            else if(ratio>low)
            {
                //weak edge
                newVal = (int)(high*256);
            }
            else
            {
                //non-relevant edge
                newVal = 0;
            }
            grad[(width*i+j)*comps]= newVal;
            grad[(width*i+j)*comps+1]= newVal;
            grad[(width*i+j)*comps+2]= newVal;
            grad[(width*i+j)*comps+3]= 255;
      	}
    }
    stbi_write_png("./pictures/DoubleTreshold.png", width, height, comps, grad, width * comps);
	
    //hysteresis
    unsigned char* cannyResult = (unsigned char*)malloc(width*height*comps);
    
    memcpy(cannyResult,grad,width*height*comps);
    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            // check neighbors for strong edges if edge is weak
            if (grad[(width * i + j) * comps] != 0 )
            {
                int newVal = 0;
                int counter=0;
                for (int x = -1; x <= 1; x++)
                {
                    
                    for (int y = -1; y <= 1; y++)
                    {
                        
                        if ((x!=0 || y!=0) && cannyResult[(width * (i + y) + j + x) * comps] == 255)
                        {
                            counter++;
                        }
                    }
                }
                //at least 1 strong pixel in radius (including itself)
                if(counter>0)
                {
                    newVal=255;
                }
                cannyResult[(width * i + j) * comps] = newVal;
                cannyResult[(width * i + j) * comps + 1] = newVal;
                cannyResult[(width * i + j) * comps + 2] = newVal;
            }
            
        }
    }
    //second pass (bottom to top)

    for (int i = height-1; i >0; i--)
    {
        for (int j = 1; j < width - 1; j++)
        {
            // check neighbors for strong edges if edge is weak
            if (grad[(width * i + j) * comps] != 0 )
            {
                int newVal = 0;
                int counter=0;
                for (int x = -1; x <= 1; x++)
                {
                    
                    for (int y = -1; y <= 1; y++)
                    {
                        
                        if ((x!=0 || y!=0) && cannyResult[(width * (i + y) + j + x) * comps] == 255)
                        {
                            counter++;
                        }
                    }
                }
                //at least 1 strong pixel in radius (including itself)
                if(counter>0)
                {
                    newVal=255;
                }
                cannyResult[(width * i + j) * comps] = newVal;
                cannyResult[(width * i + j) * comps + 1] = newVal;
                cannyResult[(width * i + j) * comps + 2] = newVal;
            }
            
        }
    }
    //third pass (right to left)
    for (int i = height-1; i >0; i--)
    {
        for (int j = width-1; j >0; j--)
        {
            // check neighbors for strong edges if edge is weak
            if (grad[(width * i + j) * comps] != 0 )
            {
                int newVal = 0;
                int counter=0;
                for (int x = -1; x <= 1; x++)
                {
                    
                    for (int y = -1; y <= 1; y++)
                    {
                        
                        if ((x!=0 || y!=0) && cannyResult[(width * (i + y) + j + x) * comps] == 255)
                        {
                            counter++;
                        }
                    }
                }
                //at least 1 strong pixel in radius (including itself)
                if(counter>0)
                {
                    newVal=255;
                }
                cannyResult[(width * i + j) * comps] = newVal;
                cannyResult[(width * i + j) * comps + 1] = newVal;
                cannyResult[(width * i + j) * comps + 2] = newVal;
            }
            
        }
    }
    FILE* cannyText = fopen("Canny.txt", "w");
    for (int i=0;i<height;i++)
    {
        for (int j = 0; j < width ; j++)
        {
            if (cannyResult[(i * width  + width + j ) * comps] == 0)
            {
                fputc('0', cannyText);
            }
            else
            {
                fputc('1', cannyText);
            }
            fputc(',', cannyText);
        }
        fputc('\n', cannyText);
    }
    fclose(cannyText);
    free(denoised);
    free(grad);
    free(magnitudes);
    free(directions);
    return cannyResult;
}
unsigned char* Halftone(unsigned char* buffer, int width, int height, int comps)
{
    unsigned char* halftone = (unsigned char*)malloc(width*height*comps*4);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {

            float pixelBrightness = (float)buffer[(i * width + j) * comps] / 255.0;
            // top left pixel
            if (pixelBrightness >= 0.8)
            {
                halftone[(i * width * 4 + j * 2) * comps] = 255;
                halftone[(i * width * 4 + j * 2) * comps + 1] = 255;
                halftone[(i * width * 4 + j * 2) * comps + 2] = 255;
            }
            else
            {
                halftone[(i * width * 4 + j * 2) * comps] = 0;
                halftone[(i * width * 4 + j * 2) * comps + 1] = 0;
                halftone[(i * width * 4 + j * 2) * comps + 2] = 0;
            }
            halftone[(i * width * 4 + j * 2) * comps + 3] = 255;
            // bottom right pixel
            if (pixelBrightness >= 0.6)
            {
                halftone[(i * width * 4 + width * 2 + j * 2 + 1) * comps] = 255;
                halftone[(i * width * 4 + width * 2 + j * 2 + 1) * comps + 1] = 255;
                halftone[(i * width * 4 + width * 2 + j * 2 + 1) * comps + 2] = 255;
            }
            else
            {
                halftone[(i * width * 4 + width * 2 + j * 2 + 1) * comps] = 0;
                halftone[(i * width * 4 + width * 2 + j * 2 + 1) * comps + 1] = 0;
                halftone[(i * width * 4 + width * 2 + j * 2 + 1) * comps + 2] = 0;
            }
            halftone[(i * width * 4 + width * 2 + j * 2 + 1) * comps + 3] = 255;
            // top right pixel
            if (pixelBrightness >= 0.4)
            {
                halftone[(i * width * 4 + j * 2 + 1) * comps] = 255;
                halftone[(i * width * 4 + j * 2 + 1) * comps + 1] = 255;
                halftone[(i * width * 4 + j * 2 + 1) * comps + 2] = 255;
            }
            else
            {
                halftone[(i * width * 4 + j * 2 + 1) * comps] = 0;
                halftone[(i * width * 4 + j * 2 + 1) * comps + 1] = 0;
                halftone[(i * width * 4 + j * 2 + 1) * comps + 2] = 0;
            }
            halftone[(i * width * 4 + j * 2 + 1) * comps + 3] = 255;
            // bottom left pixel
            if (pixelBrightness >= 0.2)
            {
                halftone[(i * width * 4 + 2 * width + j * 2) * comps] = 255;
                halftone[(i * width * 4 + 2 * width + j * 2) * comps + 1] = 255;
                halftone[(i * width * 4 + 2 * width + j * 2) * comps + 2] = 255;
            }
            else
            {
                halftone[(i * width * 4 + 2 * width + j * 2) * comps] = 0;
                halftone[(i * width * 4 + 2 * width + j * 2) * comps + 1] = 0;
                halftone[(i * width * 4 + 2 * width + j * 2) * comps + 2] = 0;
            }
            halftone[(i * width * 4 + width * 2 + j * 2) * comps + 3] = 255;
        }
    }

    FILE* halftoneText = fopen("Halftone.txt", "w");
    for (int i=0;i<height*2;i++)
    {
        for (int j = 0; j < width * 2; j++)
        {
            if (halftone[(i * width * 2 + width + j * 2) * comps] == 0)
            {
                fputc('0', halftoneText);
            }
            else
            {
                fputc('1', halftoneText);
            }
            fputc(',', halftoneText);
        }
        fputc('\n', halftoneText);
    }
    fclose(halftoneText);
    return halftone;
}
unsigned char* Floyd(unsigned char* buffer, int width, int height, int comps)
{
    unsigned char* floyd = (unsigned char*)malloc(width*height*comps);
    for(int i=0;i<height;i++)
        {
            for (int j= 0; j < width ; j++)
            {
                int newValue = buffer[(i*width+j)*comps]/16;
                int error = buffer[(i*width+j)*comps] % 16;
                //spread error in buffer
                
                if (i<height-1)
                {
                    if(j>0)
                    {
                        buffer[((i+1)*width+j-1)*comps] += error*3/16;
                    }
                    buffer[((i+1)*width+j)*comps] += error*5/16;
                    if (j < width - 1)
                    {
                        buffer[(i * width + j + 1) * comps] += error * 7 / 16;
                        
                        buffer[((i + 1) * width + j + 1) * comps] += error * 1 / 16;
                        
                    }
                }
                else
                {
                    if (j < width - 1)
                    {
                        //distribute the entire error to the pixel to the right
                        buffer[(i * width + j + 1) * comps] += error ;
                    }
                }
                
                int clampedValue = clamp(newValue*16, 0,255);
                floyd[(i*width +j)*comps]=clampedValue;
                floyd[(i*width +j)*comps+1]=clampedValue;
                floyd[(i*width +j)*comps+2]=clampedValue;
                floyd[(i*width +j)*comps+3]=255;
            }
        }
    FILE* floydText= fopen("FloydSteinberg.txt", "w");
    for (int i=0;i<height;i++)
    {
        for (int j = 0; j < width ; j++)
        {
            //string s = to_string(floyd[(i*width +j)*comps] /16) ;
            fprintf(floydText,"%2d", floyd[(i*width +j)*comps] /16);
            fputc(',', floydText);
        }
        fputc('\n', floydText);
    }
    fclose(floydText);
    return floyd;
}
int main(void)
{
    std::string filepath = "./pictures/Lenna.png";
    int width, height, comps;
    int req_comps = 4;
    unsigned char * buffer = stbi_load(filepath.c_str(), &width, &height, &comps, req_comps);
    
    //grayscale
    for( int i = 0;i<width*height;i++)
    {
        int avg = (buffer[i*comps+0]* 0.2989+ buffer[i*comps+1]* 0.5870 + buffer[i*comps+2]* 0.1140);
        avg = clamp(avg, 0, 255);
        
        buffer[i*comps+0] = avg ;
        buffer[i*comps+1] = avg ;
        buffer[i*comps+2] = avg ;
    }
    int result = stbi_write_png("./pictures/Grayscale.png", width, height, req_comps, buffer, width * comps);
    std::cout << result << std::endl;
    //canny
	unsigned char* canny = Canny(buffer,width,height,comps, 0.22f, 0.66f);
    result = stbi_write_png("./pictures/Canny.png", width, height, req_comps, canny, width * comps);
	std::cout << result << std::endl;
    //Halftone
    
    unsigned char* halftone = Halftone(buffer,width,height,comps);
    result = stbi_write_png("./pictures/Halftone.png", width*2, height*2, req_comps, halftone, width *2* comps);
	std::cout << result << std::endl;

    unsigned char* floyd = Floyd(buffer,width,height,comps);
    result = stbi_write_png("./pictures/FloydSteinberg.png", width, height, req_comps, floyd, width * comps);
    std::cout << result << std::endl;
    free(buffer);
    free(canny);
    free(halftone);
    free(floyd);
	
    return 0;
}

