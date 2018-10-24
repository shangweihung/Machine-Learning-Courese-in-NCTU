function out=rectified(input)
    out=zeros(length(input),1);
    for i=1:length(input)
        if input(i)>0
            out(i)=input(i);
        else
            out(i)=0;
        end
    end
end