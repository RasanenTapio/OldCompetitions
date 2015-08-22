# Some examples from documentation.

# Example lag code
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import io

# Generate test data:

content = '''
Time       A_x       A_y       A_z       B_x       B_y       B_z
-0.075509 -0.123527 -0.547239 -0.453707 -0.969796  0.248761  1.369613
-0.206369 -0.112098 -1.122609  0.218538 -0.878985  0.566872 -1.048862
-0.194552  0.818276 -1.563931  0.097377  1.641384 -0.766217 -1.482096
 0.502731  0.766515 -0.650482 -0.087203 -0.089075  5.443969  0.354747
 1.411380 -2.419204 -0.882383  0.005204 -0.204358 -0.999242 -0.395236
 1.036695  1.115630  2.081825 -1.038442  0.515798 -0.060016  2.669702
 0.392943  0.226386  11.039879  0.732611 -0.073447  1.164285  1.034357
-1.253264  0.389148  0.158289  0.440282 -1.195860  0.872064  0.906377
-0.133580 -0.308314 -0.839347 -0.517989  0.652120  0.477232 -0.391767
 0.623841  0.473552  0.059428  0.726088 -0.593291 -3.186297 -0.846863
 -9.075509 -0.123527 -0.547239 -0.453707 -8.969796  0.248761  1.369613
-0.206369 -0.112098 -1.122609  0.218538 -0.878985  0.566872 -1.048862
-0.194552  0.818276 -1.563931  0.097377  1.641384 -0.766217 -1.482096
 0.502731  0.766515 -0.650482 -0.087203 -0.089075  0.443969  0.354747
 1.411380 -2.419204 -0.882383  0.005204 -0.204358 -0.999242 -0.395236
 1.036695  1.115630  0.081825 -1.038442  0.515798 -0.060016  2.669702
 0.392943  0.226386  0.039879  0.732611 -0.073447  1.164285  1.034357
-1.253264  0.389148  0.158289  0.440282 -1.195860  0.872064  0.906377
-0.133580 -0.308314 -0.839347 -0.517989  0.652120  0.477232 -0.391767
 0.623841  0.473552  0.059428  0.726088 -0.593291 -3.186297 -0.846863'''

df = pd.read_table(io.BytesIO(content), sep='\s+', header=True)
df_old = np.array(df)

#column_initials = list(df.columns.values)
#print(column_initials)
#del column_initials [0]
#print(column_initials)

# (This part is working: do after smoothing?)
#for column_initial in column_initials:
#    df[column_initial+"lag1"] = df[column_initial].diff(periods=1)
#    df[column_initial+"lag2"] = df[column_initial].diff(periods=2)
#    df[column_initial+"lag3"] = df[column_initial].diff(periods=3)
#    df[column_initial+"lag4"] = df[column_initial].diff(periods=4)
#    df[column_initial+"lag5"] = df[column_initial].diff(periods=5)
#    df[column_initial+"lag6"] = df[column_initial].diff(periods=6)
	# Replace NaN?
	# Note: Use diff for filtered series
    
#print(df)

# Create callable function for apply_along_axis or parallel processing
def transform_filt(column_t):
    freqs = [6, 30]
    b,a = butter(3,np.array(freqs)/10.0,btype=filt_type)
    return lfilter(b,a,column_t)

print("Filttering data: bandpass")
filt_type = 'bandpass'
df_new = np.array(np.apply_along_axis(transform_filt, 1, df))
filt_type = 'lowpass'
df_low = np.array(np.apply_along_axis(transform_filt, 1, df))

F = np.concatenate((df_new,df_new**2,df_low), axis=1)

print("Printing filtered data")
print(df_new)

t = np.arange(0., 20., 1.0)

plt.plot(t, df_old[:,3], 'r--', t, df_new[:,3], 'd--', t, df_low[:,3])
plt.show()