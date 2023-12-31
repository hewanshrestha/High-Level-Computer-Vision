import numpy as np

from .image_filtering import gaussderiv


def normalized_histogram(img_gray, num_bins):
  assert len(img_gray.shape) == 2, 'image dimension mismatch'
  assert img_gray.dtype == 'float', 'incorrect image type'

  ### Your code here

  # raise NotImplementedError
  hists, bins = np.histogram(img_gray.reshape(img_gray.size), num_bins, (0,255))
  hists = hists/hists.sum()

  return hists, bins


def rgb_hist(img_color, num_bins):
  assert len(img_color.shape) == 3, 'image dimension mismatch'
  assert img_color.dtype == 'float', 'incorrect image type'

  # define a 3D histogram  with "num_bins^3" number of entries
  hists = np.zeros((num_bins, num_bins, num_bins))

  # execute the loop for each pixel in the image 
  for i in range(img_color.shape[0]):
      for j in range(img_color.shape[1]):
          # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
          ### Your code here
          a,b,c = img_color[i,j]
          hists[int((a*num_bins)/256), int((b*num_bins)/256), int((c*num_bins)/256)] +=1

  # normalize the histogram such that its integral (sum) is equal 1  ?????
  ### Your code here
  # raise NotImplementedError
  hists = hists/hists.sum()

  hists = hists.reshape(hists.size)
  return hists


def rg_hist(img_color, num_bins):

  """
  compute joint histogram for r/g values
  note that r/g values should be in the range [0, 1];
  histogram should be normalized so that sum of all values equals 1

  img_color - input color image
  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
  """

  assert len(img_color.shape) == 3, 'image dimension mismatch'
  assert img_color.dtype == 'float', 'incorrect image type'

  # define a 2D histogram  with "num_bins^2" number of entries
  hists = np.zeros((num_bins, num_bins))
  
  # Your code here
  A = img_color/255  #normalize
  B = np.expand_dims(np.sum(A, axis=2), axis=2)
  img_color = A/np.concatenate((B,B,B), axis = 2)


  for i in range(img_color.shape[0]):
    for j in range(img_color.shape[1]):
        # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
        ### Your code here
        a,b,c = img_color[i,j]
        hists[int((a*num_bins)/1.001), int((b*num_bins)/1.001)] +=1


  # raise NotImplementedError
  hists = hists/hists.sum()

  hists = hists.reshape(hists.size)
  return hists



def dxdy_hist(img_gray, num_bins):

  """
  compute joint histogram of Gaussian partial derivatives of the image in x and y direction
  for sigma = 7.0, the range of derivatives is approximately [-30, 30]
  histogram should be normalized so that sum of all values equals 1
  
  img_gray - input grayvalue image
  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
  
  note: you can use the function gaussderiv from the filter exercise.
  """

  assert len(img_gray.shape) == 2, 'image dimension mismatch'
  assert img_gray.dtype == 'float', 'incorrect image type'

  # compute the first derivatives
  [imgDx, imgDy] = gaussderiv(img_gray, 7.0)

  imgDx  = imgDx + 30
  imgDy  = imgDy + 30


  # quantize derivatives to "num_bins" number of values
  # define a 2D histogram  with "num_bins^2" number of entries
  hists = np.zeros((num_bins, num_bins))

  for i in range(imgDx.shape[0]):
    for j in range(imgDx.shape[1]):
      a = imgDx[i][j]
      b = imgDy[i][j]
      hists[int((a*num_bins)/60), int((b*num_bins)/60)] +=1


  # raise NotImplementedError
  hists = hists/hists.sum()
  hists = hists.reshape(hists.size)
  return hists


def dist_chi2(x,y):
  """ Compute chi2 distance between x and y """
  # your code here    
  # raise NotImplementedError
  return np.sum((x - y)**2 / (x + y + 1e-10))


def dist_l2(x,y):
  """Compute l2 distance between x and y"""
      
  # your code here    
  # raise NotImplementedError
  return np.sqrt(np.sum((x-y)**2))


def dist_intersect(x,y):

  """Compute intersection distance between x and y. Return 1 - intersection, so that smaller values also correspond to more similart histograms"""
  
  # your code here    
  # raise NotImplementedError
  Dist = 0
  for i in range(len(x)):
    Dist = min(x[i], y[i]) + Dist
  Dist = 1-Dist
  return Dist


def get_dist_by_name(x, y, dist_name):
  
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert 'unknown distance: %s'%dist_name
    

def is_grayvalue_hist(hist_name):
  
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'
    

def get_hist_by_name(img1_gray, num_bins_gray, dist_name):
  
  if dist_name == 'grayvalue':
    return normalized_histogram(img1_gray, num_bins_gray)
  elif dist_name == 'rgb':
    return rgb_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rg':
    return rg_hist(img1_gray, num_bins_gray)
  elif dist_name == 'dxdy':
    return dxdy_hist(img1_gray, num_bins_gray)
  else:
    assert 'unknown distance: %s'%dist_name