function post_process(filename)

%%In case no argument is present supply with default value
if(nargin == 0)
    disp('Default value for input file: test.cfg')
    filename = 'test.cfg';
end

%% Read data from the file
fileID = fopen(filename);
saving_dir = 'Sim'; %Initialize the name of saving directory with the default of the code
reinit_type = 'Non_Conservative_Hyperbolic'; %Initialize the name of the reinitialization choice with the default of the code
found_dir = 0; %Flag to check if we found the directory in the configuration file
found_reinit = 0; %Flag to check if we found the reinitialization type in the configuration file
while(~feof(fileID) && (found_dir == 0 || found_reinit == 0))
    line = fgetl(fileID);
    if(contains(line,'Saving_Directory'))
        found_dir = 1;
        saving_dir = line(20:end);
    elseif(contains(line,'Reinit_Type'))
        found_reinit = 1;
        reinit_type = line(15:end);
    end
end
fclose(fileID);

%% Load data from the simulation after having found the location
file = [pwd(),'/',saving_dir,'/benchmark_series.dat'];
data = load(file);
if(strcmp(reinit_type,'Non_Conservative_Hyperbolic'))
    assert(mod(numel(data), 8) == 0, 'Wrong number of values in the file')
    t = data(1:8:end);
    Vol= data(2:8:end);
    chi = data(3:8:end);
    Xc = data(4:8:end);
    Yc = data(5:8:end);
    Uc = data(6:8:end);
    Vc = data(7:8:end);
    L2_grad_phi = data(8:8:end);
elseif(strcmp(reinit_type,'Conservative'))
    assert(mod(numel(data), 7) == 0, 'Wrong number of values in the file')
    t = data(1:7:end);
    Vol = data(2:7:end);
    chi = data(3:7:end);
    Xc = data(4:7:end);
    Yc = data(5:7:end);
    Uc = data(6:7:end);
    Vc = data(7:7:end);
end

%%Plot volume behaviour
figure()
plot(t,Vol)
xlabel('t')
ylabel('Area')
title('Area evolution in time')

%%Plot circularity behaviour
figure()
plot(t,chi)
xlabel('t')
ylabel('Circularity')
title('Degree of circularity evolution in time')

%%Plot horizontal component for centroid and rising velocity
figure()
subplot(1,2,1)
plot(t,Xc)
xlabel('t')
ylabel('x_c')
title('Time evolution of x_c coordinate of the centroid')
subplot(1,2,2)
plot(t,Uc)
xlabel('t')
ylabel('u_c')
title('Time evolution of u_c coordinate of the rising velocity')

%%Plot vertical component for centroid and rising velocity
figure()
subplot(1,2,1)
plot(t,Yc)
xlabel('t')
ylabel('y_c')
title('Time evolution of y_c coordinate of the centroid')
subplot(1,2,2)
plot(t,Vc)
xlabel('t')
ylabel('v_c')
title('Time evolution of v_c coordinate of the rising velocity')

%%Plot grad_phi if necessary
if(strcmp(reinit_type, 'Non_Conservative_Hyperbolic'))
    figure()
    plot(t,L2_grad_phi)
    xlabel('t')
    ylabel('grad(\phi)')
    title('Time evolution of average grad(\phi)')
end

end
