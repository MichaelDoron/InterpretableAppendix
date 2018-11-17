from panda3d import *
from panda3d.core import *
from direct.interval.IntervalGlobal import *
from panda3d.ode import OdeWorld, OdeBody, OdeMass
from panda3d.core import Filename

from direct.showbase.ShowBase import ShowBase
 
import pandas as pd

class world(ShowBase):
 
    def __init__(self):

        ShowBase.__init__(self)

        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((.55, .55, .55, 1))
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(LVector3(0, 0, -1))
        directionalLight.setColor((0.375, 0.375, 0.375, 1))
        directionalLight.setSpecularColor((1, 1, 1, 1))

        self.ball = loader.loadModel("ball.egg")
        self.ball.reparentTo(render)
        self.ball.setPos(0,0,0)
        self.ball.setLight(render.attachNewNode(ambientLight))
        self.ball.setLight(render.attachNewNode(directionalLight))
        m = Material()
        m.setSpecular((1, 1, 1, 1))
        m.setShininess(96)
        self.ball.setMaterial(m, 1)

        duration = 5
        ball_start_position = Point3(70,(np.random.rand() * 8) - 4,-1)
        ball_velocity = Point3(0,(np.random.rand() * 8) - 4,(np.random.rand() * 8) + 6)
        self.trajectory = ProjectileInterval(self.ball,
                                             startPos = ball_start_position,
                                             startVel = ball_velocity, duration = duration)
        world = OdeWorld()
        world.setGravity(0, 0, -9.81)
        body = OdeBody(world)
        M = OdeMass()
        M.setSphere(7874, 1.0)
        body.setMass(M)
        body.setPosition(self.ball.getPos(render))
        body.setQuaternion(self.ball.getQuat(render))

        base.disableMouse()
        base.camera.setPos(80, 0, 0)
        base.camera.lookAt(0, 0, 0)
        
        base.setBackgroundColor(1,1,1)

        self.trajectory.start()

    def get_ball_pos(self):
        return self.ball.getPos().y, self.ball.getPos().z


ball_pos = []
world_nums = []
steps = []
frame_img = []
for world_num in range(400):
    print(world_num)
    app = world()
    taskMgr = taskMgr
    taskMgr.step()
    for step in range(20):
        # print app.get_ball_pos()
        taskMgr.step()
        base.graphicsEngine.renderFrame()
        base.win.saveScreenshot(Filename("frames/ball_{}_{}.png".format(world_num, step)))
        ball_pos.append(app.get_ball_pos())
        world_nums.append(world_num)
        steps.append(step)
        frame_img.append("frames/ball_{}_{}.png".format(world_num, step))
        if (ball_pos[-1][1] < -1) or (ball_pos[-1][0] < -3) or (ball_pos[-1][0] > 3): break # stop simulation if ball falls outside screen
    base.destroy()

data = pd.DataFrame(ball_pos)
data.columns = ['X','Y']
data['world_nums'] = world_nums
data['steps'] = steps
data['frame_img'] = frame_img

data.to_csv('ball_data.csv', sep=',',index=False)


bottomLeft = (app.a2dBottomLeft.get_x(), app.a2dBottomLeft.get_z())
bottomRight = (app.a2dBottomRight.get_x(), app.a2dBottomRight.get_z())
topLeft = (app.a2dTopLeft.get_x(), app.a2dTopLeft.get_z())
topRight = (app.a2dTopRight.get_x(), app.a2dTopRight.get_z())

(bottomLeft, bottomRight, topLeft, topRight)