tic
files = dir('M:\Desktop\Pattern Project\dictionary_processing\*.txt');
len = length(files);
data{len} = [];
for k = 1:len
data{k} =  textread(files(k).name, '%s', 'delimiter', ' ');
if (k == 1)
    temp = data{k};
else
    temp = vertcat(temp,data{k});
end
end
[uv,~,idx] = unique(temp);
n = accumarray(idx(:),1);
[value,index]=sort(n(:),'descend');
words = uv(index);
count = num2cell(value);
dictionary = [words,count];
toc