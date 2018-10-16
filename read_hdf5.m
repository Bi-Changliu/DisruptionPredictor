function [Time, output] = read_hdf5(device,shot_number,channel )
% author:huangyao
% email: hyao666@foxmail.com
dir_name = fix(shot_number / 200) * 200;
hl2a = {'HL-2A', 'HL2A', '2A'};
jtext = {'J-TEXT', 'JTEXT'};
east = {'EAST'};
device = upper(device);

Time=0;
output=0;
% ¼ÓÔØhdf5ÎÄ¼þ
hinfo = false;
try    
    if strcmpi(device,hl2a(1)) || strcmpi(device, hl2a(2)) || strcmpi(device,hl2a(3))
        hinfo = hdf5info(['\\192.168.9.242\hdf\2A/',num2str(dir_name),'/HL-2A=',num2str(shot_number),'=PhysicsDB.H5']);
    elseif strcmpi(device,jtext(1)) || strcmpi(device, jtext(2)) 
        hinfo = hdf5info(['\\192.168.9.242\hdf\J-TEXT/',num2str(dir_name),'/JTEXT=',num2str(shot_number),'=PhysicsDB.h5']);
    elseif strcmpi(device,east)
        hinfo = hdf5info(['\\192.168.9.242\hdf\EAST/',num2str(dir_name),'/EAST=',num2str(shot_number),'=PhysicsDB.h5']);
    end
    
catch 
    disp('No such file or directory')
    return 
end

if ~hinfo
    disp('please input right device name:')
    disp('such as:2a,jtext,east....')
    return
end
    

Data=hinfo.GroupHierarchy.Groups;

end1 = size(Data);
break_mark=false;
for index1 = 1:end1(2)
    Groups=Data(index1).Datasets;
    end2 = size(Groups);
    
    for index2 =1:end2(2)
%         Datasets=Groups(index2);
        name= Groups(index2).Name;
        
        if strfind(name,channel)
            break_mark = true;
            break
        end
        
    end
    
    if break_mark
        break
    end
end

if ~break_mark
    disp('can not find channel')
    return
end
clear end1 end2 break_mark name Groups Datasets

Data=hinfo.GroupHierarchy.Groups(index1).Datasets(index2);
output=hdf5read(Data);

data_size=size(Data.Attributes);
for index=1:data_size(2)
    name = Data.Attributes(index).Name;
    if strfind(name,'T_Start')
        T_Start = Data.Attributes(index).Value;
    elseif strfind(name,'T_Freq')
        T_Freq = Data.Attributes(index).Value;
    end
end


data_point_num = size(output);
data_point_num=data_point_num(1);
Time = (T_Start:data_point_num-1) / double(T_Freq);
Time=Time';
return
