function show_results(DATAs, DATA_clean, min_I, max_I, method_name, en_list, band2show, num_bands)
%% Use to show the video result of TC methods

numLine = ceil((length(en_list)+1)/5);
close all;

figure('units', 'normalized', 'position', [0.05, 0.5 - 0.35*numLine/2, 0.6, 0.35*numLine]);
sld = uicontrol('Style', 'slider',...
    'Min', 1, 'Max', num_bands, 'Value', band2show,...
    'Position', [220 20 1000 20],...
    'Callback', @the_fram, ...
    'SliderStep', [1/(num_bands - 1) 1/(num_bands - 1)]);
show_show(band2show)

    function show_show(band2show)
        txt = uicontrol('Style', 'text',...
            'Position', [220 45 500 20],...
            'String', append('Showing the ', num2str(band2show), 'th fram'));
        set(txt, 'Fontsize', 13)
        
        num_col_max = 5;
        
        num_col = min(num_col_max, length(en_list) + 1);
        numLine = ceil((length(en_list) + 1)/5);

        subplot(numLine, num_col, 1); imshow((DATA_clean(:, :, band2show) - min_I)/(max_I - min_I)); title('Clean');
        for i = 1:(num_col - 1)
            subplot(numLine, num_col, i+1);
            imshow((DATAs{en_list(i)}(:, :, band2show) - min_I)/(max_I - min_I)); title(method_name{en_list(i)});
        end
    end


    function the_fram(source, callbackdata)
        band2show =  round(source.Value);
        show_show(band2show)
    end
end