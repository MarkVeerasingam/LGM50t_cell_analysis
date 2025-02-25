%% Setup and Initialization
clear; clc;

% Constant temperature in degrees Celsius
T_const = 25; 

% Number of pulse files (each corresponding to a 0.05 change in SoC)
numPulses = 10;

% Sliding window parameters (in number of samples; adjust these as needed)
windowSize = 20;   % window length in samples
stepSize   = 10;   % slide step

% Initialize array to store LUT entries.
% Columns: Temperature [degC], Average Current [A], Normalized SoC, R0 [Ohm], R1 [Ohm], C1 [F]
lutEntries = [];

% Optimization options for parameter estimation
opts = optimoptions('lsqnonlin', 'Display', 'off');

%% Process each pulse file
for i = 0:(numPulses-1)
    % Construct filename for current pulse
    filename = sprintf('hppc_lut/G1/battery_G1_cycle_2_pulse_%d_lut.csv', i);
    
    % Read CSV data (assumes headers: Time, Voltage, Current)
    data = readtable(filename);
    time = data.Time;
    voltage_meas = data.Voltage;
    current = data.Current;
    
    % --- Compute a Dynamic SoC ---
    % Here we integrate current (coulomb counting) over time. This raw SoC may not span the full 0–1 range.
    % We then normalize the profile so that the minimum becomes 0 and maximum becomes 1.
    rawSOC = cumtrapz(time, current);  
    normSOC = (rawSOC - min(rawSOC)) / (max(rawSOC) - min(rawSOC));
    
    % --- Slide a window across the pulse to estimate local ECM parameters ---
    nSamples = length(time);
    for j = 1:stepSize:(nSamples - windowSize + 1)
        idx = j:(j+windowSize-1);
        
        % Data for current window
        windowTime = time(idx);
        windowVoltage = voltage_meas(idx);
        windowCurrent = current(idx);
        windowSOC = normSOC(idx);
        
        % Initial guess for ECM parameters [R0, R1, C1]
        x0 = [0.01, 0.005, 500];
        
        % Use lsqnonlin to fit the ECM model to the window data.
        % The residual function computes the difference between simulated and measured voltage.
        x_opt = lsqnonlin(@(x) ecModelResidual(x, windowTime, windowCurrent, windowVoltage), x0, [], [], opts);
        
        % Compute average SOC and average current for this window
        avgSOC = mean(windowSOC);
        avgCurrent = mean(windowCurrent);
        
        % Append a new LUT row: constant temperature, average current, average normalized SOC, and the estimated ECM parameters.
        lutEntries = [lutEntries; T_const, avgCurrent, avgSOC, x_opt(1), x_opt(2), x_opt(3)];
    end
end

%% Create and Save the Lookup Table (LUT)
lut_table = array2table(lutEntries, 'VariableNames', {'Temperature_degC', 'Current_A', 'SoC', 'R0_Ohm', 'R1_Ohm', 'C1_F'});
save('ECM_LUT_dynamic.mat', 'lut_table');

% Display the first few rows of the LUT for verification.
disp(head(lut_table));

%% --- ECM Model Residual Function ---
function res = ecModelResidual(x, time, current, meas_voltage)
    % x: parameter vector [R0, R1, C1]
    R0 = x(1);
    R1 = x(2);
    C1 = x(3);
    
    % Assume a constant open-circuit voltage (OCV). Modify this if you have an OCV-SOC relationship.
    OCV = 3.7;  
    
    % Use an average time step (assumes uniform sampling)
    dt = mean(diff(time));
    sim_voltage = zeros(size(time));
    V_RC = 0;  % Initialize the RC branch voltage
    
    % Simulate battery voltage using the ECM:
    % V(t) = OCV - I(t)*R0 - V_RC(t)
    % where dV_RC/dt = -1/(R1*C1)*V_RC + (1/C1)*I(t)
    for k = 1:length(time)
        if k > 1
            % Analytical discrete update for the RC branch using exponential decay
            alpha = exp(-dt/(R1*C1));
            V_RC = V_RC*alpha + R1*(1 - alpha)*current(k-1);
        end
        sim_voltage(k) = OCV - current(k)*R0 - V_RC;
    end
    
    % Return the difference between simulated and measured voltage.
    res = sim_voltage - meas_voltage;
end
