t  =load('debug.txt');
[B,A] = butter(2,0.1);
%t = filtfilt(B,A,t); 
plot(t)
hold on