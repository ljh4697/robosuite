import trimesh
import os
from lxml import etree




mesh = trimesh.load('/home/joonhyeok/robosuite/robosuite/models/assets/objects/meshes/human.obj', force='mesh')
#convex_list = mesh.convex_decomposition(maxhulls=20, resolution=8000000, oclAcceleration=1)

mesh_file = "/home/joonhyeok/robosuite/robosuite/models/assets/objects/"
mesh_type = "human"


mesh.apply_translation(-mesh.center_mass)
unit = 0.001


if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(tuple(trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)))
    for g in mesh.geometry.values(): # clean mesh vertices and surfaces        
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()
        mesh.remove_duplicate_faces()



convex_list = mesh.convex_decomposition(maxhulls=20, resolution=8000000, oclAcceleration=1) # output 으로 decomposition 한 여러개의 mesh 의 array 나옴


for i in range(0, len(convex_list)):
    savename = mesh_file + 'meshes/' + mesh_type + '_' + str(i) + '.stl'
    part_trimesh = convex_list[i]

    if not os.path.exists(os.path.dirname(savename)):                    
        os.makedirs(os.path.dirname(savename))                
        trimesh.exchange.export.export_mesh(part_trimesh, savename, file_type="stl")        
    else:
        trimesh.exchange.export.export_mesh(part_trimesh, savename, file_type="stl")        
        

savename = mesh_file + 'meshes/' + mesh_type + '.stl'

if not os.path.exists(os.path.dirname(savename)):                
    os.makedirs(os.path.dirname(savename))
    trimesh.exchange.export.export_mesh(mesh, savename, file_type="stl")
else:
    trimesh.exchange.export.export_mesh(mesh, savename, file_type="stl") 
    
    
mujoco = etree.Element('mujoco', model=mesh_type)        
asset = etree.Element('asset')        
worldbody = etree.Element('worldbody')        
body = etree.Element('body', name=mesh_type)        
body_col = etree.Element('body', name='collision')        
body_vis = etree.Element('body', name='visual')        
site1 = etree.Element('site', rgba='0 0 0 0', size='0.005', pos='0 0 -0.06', name='bottom_site')        
site2 = etree.Element('site', rgba='0 0 0 0', size='0.005', pos='0 0 0.04', name='top_site')        
site3 = etree.Element('site', rgba='0 0 0 0', size='0.005', pos='0.025 0.025 0', name='horizontal_radius_site')

if not isinstance(convex_list, trimesh.base.Trimesh):            
    for i in range(0, len(convex_list)):
        asset.append(etree.Element('mesh', file='meshes/' + mesh_type + '_' + str(i) + '.stl',       


        name=mesh_type + '_' + str(i), scale=str(unit)+' '+str(unit)+' '+str(unit)))        

        body_col.append(etree.Element('geom', pos='0 0 0', mesh=mesh_type + '_' + str(i), type='mesh',                
                                    friction='1 0.005 0.0001',                
                                    solimp='1. 1. 0.001 0.02 6',           
                                    solref='0.02 1',               
                                    density='5000', rgba='0 1 0 1',              
                                    group='1', condim='6')) 
                                                
        body_col.append(etree.Element('geom', pos='0 0 0', mesh=mesh_type + '_' + str(i), type='mesh',
                                    density='5000', rgba='0 1 0 1',                                              group='1', condim='6'))
                        
                        
        body_vis.append(etree.Element('geom', pos='0 0 0', mesh=mesh_type + '_' + str(i), type='mesh', rgba='0 1 0 1',
                                    conaffinity='0', contype='0', group='0', mass='0.0001'))

else:
    asset.append(etree.Element('mesh', file='meshes/' + mesh_type + '.stl',
                                        name=mesh_type, scale=str(unit)+' '+str(unit)+' '+str(unit)))

    body_col.append(etree.Element('geom', pos='0 0 0', mesh=mesh_type, type='mesh',           
                            fiction='1 0.005 0.0001',           
                            solimp='1. 1. 0.001 0.02 6',           
                            solref='0.02 1',                                        
                            density='5000', rgba='0 1 0 1',
                            group='1', condim='6'))

    body_col.append(etree.Element('geom', pos='0 0 0', mesh=mesh_type, type='mesh',
                            density='5000', rgba='0 1 0 1',
                            group='1', condim='6'))
            
    body_vis.append(etree.Element('geom', pos='0 0 0', mesh=mesh_type, type='mesh', rgba='0 1 0 1',                              
                            conaffinity='0', contype='0', group='0', mass='0.0001'))



mujoco.append(worldbody)        
mujoco.append(asset)        
body.append(body_col)        
body.append(body_vis)        
body.append(site1)        
body.append(site2)        
body.append(site3)        
worldbody.append(body)        
s = etree.tostring(mujoco, pretty_print=True)        
file = open(mesh_file + mesh_type + '.xml', 'wb')        
file.write(s)        
file.close()

print(mesh_type + 'processing completed')

print(convex_list)

