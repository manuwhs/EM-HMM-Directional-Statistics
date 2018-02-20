% Assumes that you have the following variables
% - topomaps: matrix of size (# channels x # states)
% - Z: vetor, assignment of each time-point to a state (length T*SUBJECTS)
%		NB! Here it is concatenated for all subjects to begin with
% - T: number, length of each subjects data (assumed to be equal across subjects)
% - time: a vector specifying the timing relative to trial presentation (length T)
% - chanlocs: EEGLAB chanlocs

addpath(PATH_TO_EEGLAB)

out_file = 'facescram';

% subject by state-time course matrix
Zprsub = nan(SUBJECTS,T);
for su = 1:SUBJECTS
   Zprsub(su,:) = Z( (1:T)+(su-1)*T);
end


% create colormap
cmap = distinguishable_colors(K);

%% Plot stuff

% topographical maps
paperpos = [0 0 14.5 10];
FIG=figure;
for k = 1:K
   subplot(2,5,k) 
   topoplot(topomaps(:,k), chanlocs)
   title( sprintf('State %i',k), 'Color', cmap(k,:) )
end

set(FIG,'PaperUnits', 'centimeters');
set(FIG, 'PaperPosition', paperpos);
print(FIG, '-depsc', '-r600', sprintf('figs/%s_topomaps',out_file))
print(FIG, '-dpng', '-r1000', sprintf('figs/%s_topomaps',out_file))

% State-Time Image
paperpos = [0 0 14.5 5];
FIG = figure;
image(Zprsub); 
CBAR = colorbar; CBAR.Ticks = (1:K) + 0.5; CBAR.TickLabels = 1:K;
colormap(cmap)
desired_xticks = [-0.1,0,0.2,0.4]; 
act_xticks = interp1(time,1:T, desired_xticks ); act_xticks = round(act_xticks);
act_xticks(1) = 1;
set(gca,'XTick',act_xticks)
set(gca,'XTickLabel',desired_xticks)
ylabel('Training subjects', 'FontSize',fs)
xlabel('Time after stimulus [s]','FontSize',fs)
set(FIG,'PaperUnits', 'centimeters');
set(FIG, 'PaperPosition', paperpos);
print(FIG, '-depsc', '-r600', sprintf('figs/%s_statetime',out_file))
print(FIG, '-dpdf', '-r600', sprintf('figs/%s_statetime',out_file))
print(FIG, '-dpng', '-r1000', sprintf('figs/%s_statetime',out_file))