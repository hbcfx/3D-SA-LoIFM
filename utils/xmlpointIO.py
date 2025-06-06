import xml.dom.minidom

def pointxml(points_container,outputfilename):

    doc = xml.dom.minidom.Document()
    root = doc.createElement('point_set_file')
    doc.appendChild(root)

    file_version = doc.createElement('file_version')
    file_version.appendChild(doc.createTextNode('0.1'))
    root.appendChild(file_version)

    point_set = doc.createElement('point_set')

    time_series = doc.createElement('time_series')

    time_series_id = doc.createElement('time_series_id')
    time_series_id.appendChild(doc.createTextNode('0'))
    time_series.appendChild(time_series_id)

    geometry3D=doc.createElement('Geometry3D')
    geometry3D.setAttribute('ImageGeometry','True')
    geometry3D.setAttribute('FrameOfReferenceID','0')

    indexToWorld = doc.createElement('IndexToWorld')
    indexToWorld.setAttribute('type','Matrix3x3')
    indexToWorld.setAttribute('m_0_0',"1")
    indexToWorld.setAttribute('m_0_1',"0")
    indexToWorld.setAttribute('m_0_2',"0")

    indexToWorld.setAttribute('m_1_0', "0")
    indexToWorld.setAttribute('m_1_1', "1")
    indexToWorld.setAttribute('m_1_2', "0")

    indexToWorld.setAttribute('m_2_0', "0")
    indexToWorld.setAttribute('m_2_1', "0")
    indexToWorld.setAttribute('m_2_2', "1")

    geometry3D.appendChild(indexToWorld)

    offset = doc.createElement('Offset')
    offset.setAttribute('type', "Vector3D")
    offset.setAttribute('x', "0")
    offset.setAttribute('y', "0")
    offset.setAttribute('z', "0")
    geometry3D.appendChild(offset)

    bounds = doc.createElement('Bounds')
    min = doc.createElement('Min')
    min.setAttribute('type','Vector3D')
    #min.setAttribute('x',landmark_num[3])
    #min.setAttribute('y',landmark_num[4])
    #min.setAttribute('z',landmark_num[-1])

    max = doc.createElement('Max')
    max.setAttribute('type', 'Vector3D')
   
    time_series.appendChild(geometry3D)

    minVal=[points_container[0][0].cpu(),points_container[0][1].cpu(),points_container[0][2].cpu().item()]
    maxVal=[points_container[0][0].cpu(),points_container[0][1].cpu(),points_container[0][2].cpu().item()]

    for pn in range(len(points_container)):



        point = doc.createElement('point')
        id = doc.createElement('id')
        id.appendChild(doc.createTextNode(str(pn)))

        specification = doc.createElement('specification')
        specification.appendChild(doc.createTextNode('0'))

        x = doc.createElement('x')
        #   x_num=(float(landmark_num[2])-origin[0])*spcaing[0]
        x.appendChild(doc.createTextNode(str(points_container[pn][0].cpu().item())))

        for dim in range(len(points_container[pn])):

            if minVal[dim]>points_container[pn][dim].cpu().item():
                minVal[dim]=points_container[pn][dim].cpu().item()

            if maxVal[dim]<points_container[pn][dim].cpu().item():
                maxVal[dim]=points_container[pn][dim].cpu().item()   

        y = doc.createElement('y')
        #y_num=(float(landmark_num[3])-origin[1])*spcaing[1]
        y.appendChild(doc.createTextNode(str(points_container[pn][1].cpu().item())))

        z = doc.createElement('z')
        #z_num=(float(landmark_num[4])-origin[2])*spcaing[2]
        z.appendChild(doc.createTextNode(str(points_container[pn][2].cpu().item())))

        point.appendChild(id)
        point.appendChild(specification)
        point.appendChild(x)
        point.appendChild(y)
        point.appendChild(z)

        time_series.appendChild(point)

    min.setAttribute('x', str(minVal[0]))
    min.setAttribute('y', str(minVal[1]))
    min.setAttribute('z', str(minVal[2]))
    

    max.setAttribute('x', str(maxVal[0]))
    max.setAttribute('y', str(maxVal[1]))
    max.setAttribute('z', str(maxVal[2]))

    bounds.appendChild(min)
    bounds.appendChild(max)
    geometry3D.appendChild(bounds)


    point_set.appendChild(time_series)

    root.appendChild(point_set)

        # 此处需要用codecs.open可以指定编码方式
        
    fp = open(outputfilename, 'w')
        # 将内存中的xml写入到文件
    doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding='utf-8')
    fp.close()