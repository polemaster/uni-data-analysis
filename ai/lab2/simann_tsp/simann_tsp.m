% Simulated Annealing for Travelmen Salesman Problem
clear;                         % clearing workspace
tic;
number_of_steps = 100000;
steps_to_show = 1000;

T = 10000;                     % initial temperature
Tmin = 0.01;
wT = 0.995;                    % temperature correction factor after successful modification
c = 0.1;                       % constant for differentiate the impact of temperature on acceptance prob. and searching area (not used here)
cycle_length_min = 10e40;      % best cycle length
cities = load('kroa10');

number_of_cities = length(cities(:,1))
N = number_of_cities;

Solution = randperm(number_of_cities);       % initial solution - random permutation of cities sequence
 
EdgeLengths = sqrt((cities(Solution,2)-cities(Solution([2:end 1]),2)).^2 + ...    % distances between cities
               (cities(Solution,3)-cities(Solution([2:end 1]),3)).^2); 
CycleLength = sum(EdgeLengths);              % length of cycle
CycleLengthPrev = CycleLength;
CycleLengthRec = []; StepRec = [];
Lengths = []; Temp = []; Paccept = [];

format long;                   % 15 significant places

RecordRoute = [Solution];
for step=1:number_of_steps
   % Changing the solution by inversion - inverting the drawn substring:
   k1 = 1+floor(rand*N);            % edge draw for replacement
   k2 = k1+floor(rand*(N-k1));      % inversion of selected substring
   SolutionNew = [Solution(1:k1) fliplr(Solution((k1+1):k2)) Solution((k2+1):end)];
   % CycleLength correction:
   CycleLength = sum(sqrt((cities(SolutionNew,2)-cities(SolutionNew([2:end 1]),2)).^2 + ...
               (cities(SolutionNew,3)-cities(SolutionNew([2:end 1]),3)).^2));
   Lengths(step) = CycleLength;
   if CycleLength < cycle_length_min             % checking if record
      disp('Improve cycle length:')
      cycle_length_min = CycleLength
      %disp('For solution:')
      SolutionMinCycleLength = SolutionNew;
      CycleLengthRec = [CycleLengthRec CycleLength];
      StepRec = [StepRec step];
   end
   dE = CycleLength - CycleLengthPrev;           % difference between solutions (if dE < 0 than new solution is better) 
   if rand < 1/(1+exp(dE/(c*T)))                 % acceptance condition
      Solution = SolutionNew;
      %RecordRoute = [RecordRoute ; Solution];
      CycleLengthPrev = CycleLength;
      Lengths = [Lengths CycleLength];
   end
   T = T*wT;
   if T<Tmin 
      T=Tmin;
   end
   Temp = [Temp T];
   Paccept = [Paccept 1/(1+exp(dE/(c*T)))];
   if mod(step,steps_to_show)==0
      %close;
      cla;
      plot(cities(SolutionMinCycleLength([1:end,1]),2),cities(SolutionMinCycleLength([1:end,1]),3));
      hold on;
      plot(cities(:,2),cities(:,3),'ro');
      title(sprintf('in %g step min. TSP cycle length = %g',step,cycle_length_min));
      hold off;
      pause(0.1);
      drawnow;
      shg;
   end
     
end

plot(Lengths);
title(' Accepted route lengths');
pause
plot(StepRec,CycleLengthRec,'*');
title(' Record route lengths ');
pause
plot(Temp);
title(' temperatures ');
pause
plot(Paccept);
title('probabilities of acceptance');
