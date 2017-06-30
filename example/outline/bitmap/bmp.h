#ifndef BMP_H_INCLUDED
#define BMP_H_INCLUDED

#include <vector>
#include <cstdio>
#include <cstdlib>

struct bmp_t
{
  char type[2];
  unsigned long int file_size;
  unsigned long int start_addr;
  unsigned long int header_size;
  unsigned long int width;
  unsigned long int height;
  unsigned long int nb_plans;
  unsigned long int nb_bit_per_pixel;
  char mask;
  unsigned long int compression_type;
  unsigned long int image_size; //nb of pixel
  unsigned long int resolution_h; //pixel/meter
  unsigned long int resolution_v; //pixel/metre
  unsigned long int nb_colors_in_palette;
  unsigned long int nb_significant_colors;
  unsigned int palette_size;
  std::vector<unsigned char> palette;
  std::vector< std::vector<unsigned long int> > image;
};

class bmp
{
public:
  bmp();
  ~bmp();

  bmp_t read_bmp(char* file_name);
  void write_bmp(char* file_name);

  bmp_t input_bmp;
  bmp_t output_bmp;

  float image_mean;
  float ecart_type;

private:
  void get_bmp_header(FILE* file);
  void generate_bmp_header(FILE* file);

  void get_image(FILE* file);
  void generate_image(FILE* file);

  void get_image_real_color(FILE* file);
  void generate_image_real_color(FILE* file);

  unsigned long int four_to_int(unsigned char data[4]);
  unsigned long int  three_to_int(unsigned char data[3]);
  unsigned long int two_to_int(unsigned char data[2]);
  void int_to_four(unsigned long int data, unsigned char converted[4]);
  void int_to_three(unsigned long int data, unsigned char converted[3]);
  void int_to_two(unsigned long int data,unsigned char converted[2]);
  char get_mask(unsigned long int nb_pixel_bit_per_pixel);
};

#endif // BMP_H_INCLUDED
