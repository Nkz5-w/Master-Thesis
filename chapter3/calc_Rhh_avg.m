num_subcarrier = 512;   % 载波数
num_cp = 32;
delta_f = 35e3;         % 子载波间隔
pathDelays = [5e-8 3e-7 6e-7];
pathGains = [-5 -6 -10];
fs = num_subcarrier * delta_f;
Rhh_avg=zeros(num_subcarrier,num_subcarrier);
modulation_mode = 2;    % 调制方式
% 生成导频序列，并进行载波调制
pilot_data=ones(log2(modulation_mode)*num_subcarrier,1);
pilot_symbols=qammod(pilot_data,modulation_mode,'InputType','bit');

for i=1:2500
    chan = comm.RayleighChannel(PathGainsOutputPort=true, ...
               SampleRate=fs,...
               MaximumDopplerShift=randi([200,800]), ...
               PathDelays=pathDelays, ...
               AveragePathGains=pathGains);
    release(chan);
    reset(chan);
    Tx_piloted_symbols_test=ofdmmod(pilot_symbols,num_subcarrier,num_cp); 
    [pilot_symbols_test,~]=chan(Tx_piloted_symbols_test);
    pilot_symbols_test=awgn(pilot_symbols_test,randi([10,30]),'measured');
    pilot_symbols_test=ofdmdemod(pilot_symbols_test,num_subcarrier,num_cp);
    h=pilot_symbols_test./pilot_symbols;
    Rhh_temp=h*h';
    Rhh_avg=Rhh_avg+(Rhh_temp-Rhh_avg)/i;
end

% 保存结果
save('Rhh_avg_data.mat', 'Rhh_avg');
fprintf('Rhh_avg 计算完成并已保存\n');