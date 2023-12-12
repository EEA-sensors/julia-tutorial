


%names = {'C++','Java','Julia','Matlab','Octave','Python'};
%f_times = [0.256905 1.699 4.287048136 13.968916 60.6021 123.842566967];
%s_times = [0.490346 1.543 4.710070576 21.610301 87.7424 103.221923113];

%names = {'C++','Julia','Matlab','Python'};
%f_times = [0.157409 1.743044833 3.683195 31.48172116279602];
%s_times = [0.247047 1.862426958 7.447597 26.56629204750061];

%names = {'C++','Julia','Matlab','Python','JAX'};
%f_times = [0.127301 0.055515 3.699335 14.52953314781189 1.2923650741577148];
%s_times = [0.215421 0.118204 7.385088 14.896209001541138 1.4345130920410156];

names = {'C++','Julia','Matlab','Python','JAX','TensorFlow'};
f_times = [0.127301 0.055515 3.699335 14.52953314781189 1.2923650741577148 48.27635407447815];
s_times = [0.215421 0.118204 7.385088 14.896209001541138 1.4345130920410156 41.564396142959595];


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
    set(gca,'FontSize',12)

    th = [];
    lbl = [];
    for i=1:numel(y), % create annotations
      % store the handles for later use
      th(i) = text(y(i)+15,x(i),sprintf('%.1f',y(i))) ;
      lbl(i) = text(-10,x(i),names{i}) ;
    end
    set(gca,'YTick',[])
    
    xlabel('Elapsed time in seconds');

    % center all annotations at once using the handles
    set(th,'horizontalalignment','center', ...
    'verticalalignment','bottom','FontSize',12) ; 

    set(lbl,'horizontalalignment','right', ...
    'verticalalignment','bottom','FontSize',12) ; 

    title(titl);
end

rel_tot = f_times + s_times;
rel_tot = rel_tot / rel_tot(2);

for i=1:length(names)
    fprintf('%-7s ',names{i});
    if i < length(names)
        fprintf('& ');
    else
        fprintf('\\\\ \n');
    end
end

for i=1:length(names)
    fprintf('%-7.1f ',rel_tot(i));
    if i < length(names)
        fprintf('& ');
    else
        fprintf('\\\\ \n');
    end
end

print -dpng -r300 times_2023;

