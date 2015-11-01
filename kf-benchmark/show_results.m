


names = {'C++','Java','Julia','Matlab','Octave','Python'};
f_times = [0.256905 1.699 8.960016802 13.968916 60.6021 138.077860117];
s_times = [0.490346 1.543 12.547804952 21.610301 87.7424 107.945866108];

clf;


for i=1:2
    subplot(1,2,i);
    
    x = 1:length(names);
    if i == 1
        y = f_times;
        titl = 'Kalman filter';
    else
        y = s_times;
        titl = 'RTS smoother';
    end

    h = barh(x,y) ; % create bar
    set(h,'FaceColor',[1 1 0]);

    th = [];
    lbl = [];
    for i=1:numel(y), % create annotations
      % store the handles for later use
      th(i) = text(y(i)+6,x(i),sprintf('%.1f',y(i))) ;
      lbl(i) = text(-15,x(i),names{i}) ;
    end
    set(gca,'YTick',[])
    xlabel('Elapsed time in seconds');

    % center all annotations at once using the handles
    set(th,'horizontalalignment','center', ...
    'verticalalignment','bottom') ; 

    title(titl);
end

rel_tot = f_times + s_times;
rel_tot = rel_tot / rel_tot(1);

for i=1:length(names)
    fprintf('%-7s ',names{i});
    if i < length(names)
        fprintf('& ');
    else
        fprintf('\\\\ \n');
    end
end

for i=1:length(names)
    fprintf('%-7d ',round(rel_tot(i)));
    if i < length(names)
        fprintf('& ');
    else
        fprintf('\\\\ \n');
    end
end

