

# import basic libraries
import os
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

# import message declarations
from Basilisk.architecture import messaging

# import FSW Algorithm related support
from Basilisk.fswAlgorithms import locationPointing
from Basilisk.fswAlgorithms import mrpFeedback

# import simulation related support (especially stripLocation)
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import stripLocation
from Basilisk.simulation import simpleNav
from Basilisk.simulation import spacecraft

# import general simulation support files
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import astroFunctions
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import (
    unitTestSupport,
)  # general support file with common unit test functions

# attempt to import vizard
from Basilisk.utilities import vizSupport

# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from scipy.spatial.transform import Rotation
bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

# # Plotting functions

def transform_body_to_inertial(mrp, body_vector):
    """
    Transforms a vector from the body frame to the inertial frame using the spacecraft's MRP attitude.
    
    Parameters:
    - mrp: A 3-element array representing the spacecraft's MRP vector.
    - body_vector: A 3-element array representing the vector in the body frame to be transformed.
    - spacecraft_position: A 3-element array representing the spacecraft's position in the inertial frame.
    
    Returns:
    - A 3-element array representing the vector in the inertial frame.
    """
    # use scipy to convert MRP to a rotation matrix
    r = Rotation.from_mrp(mrp)
    rotation_body_to_inertial = r.as_matrix()

    # Transform the vector from body frame to inertial frame
    inertial_vector = np.dot(rotation_body_to_inertial, body_vector)

    return inertial_vector



def ray_sphere_intersection(camera_position, camera_direction, earth_radius):
    """
    Computes the intersection points between a ray (from the camera) and a sphere (Earth).
    
    Parameters:
    - camera_position: A 3-element array representing the camera's position in the inertial frame.
    - camera_direction: A 3-element array representing the unit direction vector of the ray (camera's pointing direction).
    - earth_radius: The radius of the Earth (in the same units as the position).
    
    Returns:
    - A list of intersection points (if they exist). If no intersection, returns an empty list.
    """

    
    # Vector from the camera to the Earth's center
    L = camera_position
    
    # Coefficients for the quadratic equation
    a = np.dot(camera_direction, camera_direction)
    b = 2.0 * np.dot(camera_direction, L)
    c = np.dot(L, L) - earth_radius**2
    
    # Discriminant of the quadratic equation
    discriminant = b**2 - 4 * a * c
    
    # No intersection if the discriminant is negative
    if discriminant < 0:
        return [0,0,0]
    
    # Compute the two possible solutions for t
    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)
    
    # Compute the intersection points
    intersection = camera_position + min(t1,t2) * camera_direction
    return intersection


# def plot_intersection(camera_position,camera_direction_body,earth_radius,mrp):
#     intersections=[]
#     for i in range(len(mrp)):
#         body_vector=transform_body_to_inertial(mrp[i], camera_direction_body)
#         intersections.append(ray_sphere_intersection(camera_position[i], body_vector, earth_radius))
#         #plot intersection points in 3D
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_box_aspect([1,1,1])
#         ax.set_xlabel('X [m]')
#         ax.set_ylabel('Y [m]')
#         ax.set_zlabel('Z [m]')
#         ax.set_title('Intersection Points')
#         ax.scatter(camera_direction_body[1][0], intersections[1][1], intersections[1][2], color='blue', label='Camera Position')

        


def plot_attitude_error(timeLineSet, dataSigmaBR, strip_start_time:float):
    """Plot the attitude result."""
    plt.figure(1)
    fig = plt.gcf()
    ax = fig.gca()
    vectorData = dataSigmaBR
    sNorm = np.array([np.linalg.norm(v) for v in vectorData])
    plt.plot(timeLineSet, sNorm,
             color=unitTestSupport.getLineColor(1, 3),
             )
    plt.xlabel('Time [min]')
    plt.ylabel(r'Attitude Error Norm $|\sigma_{B/R}|$')
    ax.set_yscale('log')
    # Plot the vertical dashed line
    plt.axvline(x=strip_start_time, color='black', linestyle=':', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7)
    # Add vertical text on the line
    plt.text(strip_start_time, dataSigmaBR.max() * 0.15, 'Start strip imaging', 
             rotation=90, verticalalignment='center', horizontalalignment='right', 
             color='black')
    plt.text(0, dataSigmaBR.max() * 0.15, 'Pointing to start', 
             rotation=90, verticalalignment='center', horizontalalignment='right', 
             color='black')

def plot_km_error(timeLineSet, dataSigmaBR,tran,strip_start_time:float):
    """Plot the attitude result."""
    plt.figure(6)
    fig = plt.gcf()
    ax = fig.gca()
    vectorData = dataSigmaBR
    sNorm = np.array([4*np.arctan(np.linalg.norm(vectorData[i]))*(np.linalg.norm(tran[i])-astroFunctions.E_radius * 1e3) for i in range(len(vectorData))])
    plt.plot(timeLineSet, sNorm,
             color=unitTestSupport.getLineColor(1, 3),
             )
    plt.xlabel('Time [min]')
    plt.ylabel(r'Km Error Norm $|\sigma_{B/R}|$')
    ax.set_yscale('log')
    # Plot the vertical dashed line
    plt.axvline(x=strip_start_time, color='black', linestyle=':', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7)
    # Add vertical text on the line
    plt.text(strip_start_time, dataSigmaBR.max() * 0.15, 'Start strip imaging', 
             rotation=90, verticalalignment='center', horizontalalignment='right', 
             color='black')
    plt.text(0, dataSigmaBR.max() * 0.15, 'Pointing to start', 
             rotation=90, verticalalignment='center', horizontalalignment='right', 
             color='black')


def plot_control_torque(timeLineSet, dataLr, strip_start_time:float):
    """Plot the control torque response."""
    plt.figure(2)
    for idx in range(3):
        plt.plot(timeLineSet, dataLr[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='$L_{r,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Control Torque $L_r$ [Nm]')
    # Plot the vertical dashed line
    plt.axvline(x=strip_start_time, color='black', linestyle=':', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7)
    # Add vertical text on the line
    plt.text(strip_start_time, dataLr.max() * 0.4, 'Start strip imaging', 
             rotation=90, verticalalignment='center', horizontalalignment='right', 
             color='black')
    plt.text(0, dataLr.max() * 0.4, 'Pointing to start', 
             rotation=90, verticalalignment='center', horizontalalignment='right', 
             color='black')


def plot_rate_error(timeLineSet, dataOmegaBR, strip_start_time:float):
    """Plot the body angular velocity tracking error."""
    plt.figure(3)
    for idx in range(3):
        plt.plot(timeLineSet, dataOmegaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Rate Tracking Error [rad/s] ')
        # Plot the vertical dashed line
    plt.axvline(x=strip_start_time, color='black', linestyle=':', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7)
    # Add vertical text on the line
    plt.text(strip_start_time, dataOmegaBR.max() * 0.4, 'Start strip imaging', 
             rotation=90, verticalalignment='center', horizontalalignment='right', 
             color='black')
    plt.text(0, dataOmegaBR.max() * 0.4, 'Pointing to start', 
             rotation=90, verticalalignment='center', horizontalalignment='right', 
             color='black')

def plot_access(timeLineSet, hasAccess, strip_start_time:float):
    plt.figure(4)
    plt.plot(timeLineSet, hasAccess)
    # Plot the vertical dashed line
    plt.axvline(x=strip_start_time, color='black', linestyle=':', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.7)
    # Add vertical text on the line
    plt.text(strip_start_time, max(hasAccess) * 0.4, 'Start strip imaging', 
             rotation=90, verticalalignment='center', horizontalalignment='right', 
             color='black')
    plt.text(0, max(hasAccess) * 0.4, 'Pointing to start', 
             rotation=90, verticalalignment='center', horizontalalignment='right', 
             color='black')
    plt.xlabel("Time [min]")
    plt.ylabel("Imaging Target Access")

def create_earth(ax, earth_radius):
    # Create a sphere representing the Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create a light source object for shading
    ls = LightSource(azdeg=0, altdeg=65)
    rgb = ls.shade(z, cmap=plt.cm.gray, vert_exag=0.1, blend_mode='soft')
    
    # Plot the Earth as a surface with grid lines
    ax.plot_surface(x, y, z, facecolors=rgb, rstride=5, cstride=5, alpha=0.1, edgecolor='k')
    
    # Set the axes to have the same proportion
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def set_zoom(ax, r_LP_N):
    # Set the axes to zoom in on the trajectory
    max_range = np.array([r_LP_N[:, 0].max()-r_LP_N[:, 0].min(), 
                          r_LP_N[:, 1].max()-r_LP_N[:, 1].min(), 
                          r_LP_N[:, 2].max()-r_LP_N[:, 2].min()]).max() / 2.0
    mid_x = (r_LP_N[:, 0].max()+r_LP_N[:, 0].min()) * 0.5
    mid_y = (r_LP_N[:, 1].max()+r_LP_N[:, 1].min()) * 0.5
    mid_z = (r_LP_N[:, 2].max()+r_LP_N[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def clear_previous_plot(ax):
    # Clear the previous trajectory
    for collection in ax.collections:
        collection.remove()
    for line in ax.lines:
        line.remove()

def plot_trajectory(ax, trajectory_points, current_step, camera_vectors=None, camera_n_skip=25, name='Trajectory', style='-', color='blue', color_by_iteration=False):
    if color_by_iteration:
        # Create a colormap
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, current_step))
        
        # Plot the trajectory with colors based on iteration
        for i in range(current_step):
            ax.plot(trajectory_points[i:i+2, 0], trajectory_points[i:i+2, 1], trajectory_points[i:i+2, 2], style, color=colors[i], markersize=2, linewidth=1)
    else:
        # Plot the trajectory up to the current step with circle markers
        ax.plot(trajectory_points[:current_step, 0], trajectory_points[:current_step, 1], trajectory_points[:current_step, 2], style, label=name, color=color, markersize=2, linewidth=1)
    
    # Plot the initial point with a triangle marker in red
    ax.scatter(trajectory_points[0, 0], trajectory_points[0, 1], trajectory_points[0, 2], color=color, marker='^', label=f"Initial {name}")
    
    # Always plot the final point with a square marker in green
    ax.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], trajectory_points[-1, 2], color=color, marker='s', label=f"Final {name}")
    
    if camera_vectors is not None:
        # Plot the camera vectors as arrows for every camera_n_skip points
        ax.quiver(trajectory_points[:current_step:camera_n_skip, 0], trajectory_points[:current_step:camera_n_skip, 1], trajectory_points[:current_step:camera_n_skip, 2],
                  camera_vectors[:current_step:camera_n_skip, 0], camera_vectors[:current_step:camera_n_skip, 1], camera_vectors[:current_step:camera_n_skip, 2],
                  length=astroFunctions.E_radius * 1e3 * 0.1, normalize=True, color=color, label='Camera Vectors')
    
    ax.legend()

def plot_3D_R_LP_N(ax, r_LP_N, earth_radius, tran, camera_vectors, intersections, current_step, init=False, plot_earth=True, zoom_trajectory=False):
    if init:
        if plot_earth:
            create_earth(ax, earth_radius)
        if zoom_trajectory:
            set_zoom(ax, r_LP_N)
        
        ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1
        
        # Remove the figure grid
        ax.grid(False)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Strip Boulder-Phoenix')
    
    clear_previous_plot(ax)
    
    if plot_earth:
        create_earth(ax, earth_radius)
    
    if zoom_trajectory:
        set_zoom(ax, r_LP_N)
    
    plot_trajectory(ax, r_LP_N, current_step, name='strip', style='-', color='blue')
    plot_trajectory(ax, intersections[120:], current_step, name='Intersection',style='o', color='red', color_by_iteration=True)
    plot_trajectory(ax, tran, current_step, camera_vectors=camera_vectors, name='satellite', camera_n_skip=25, style='-', color='green')

def generate_video(timeLineSet, r_LP_N, earth_radius, tran, camera_vectors, intersections, filename='trajectory.mp4', plot_earth=True, zoom_trajectory=False):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12.8, 7.2))
    
    # Initialize the plot
    plot_3D_R_LP_N(ax, r_LP_N, earth_radius, tran, camera_vectors, intersections, current_step=0, init=True, plot_earth=plot_earth, zoom_trajectory=zoom_trajectory)
    
    def update(current_step):
        plot_3D_R_LP_N(ax, r_LP_N, earth_radius, tran, camera_vectors, intersections, current_step=current_step, plot_earth=plot_earth, zoom_trajectory=zoom_trajectory)
    
    ani = animation.FuncAnimation(fig, update, frames=len(r_LP_N), repeat=False)
    ani.save(filename, writer='ffmpeg', fps=30)

def plot_full_trajectory(r_LP_N, earth_radius, tran, camera_vectors, intersections, plot_earth=True, zoom_trajectory=False):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12.8, 7.2))
    plot_3D_R_LP_N(ax, r_LP_N, earth_radius, tran, camera_vectors,intersections ,current_step=len(r_LP_N), init=True, plot_earth=plot_earth, zoom_trajectory=zoom_trajectory)
    plt.show()

def plot_error_m(timeLineSet,r_LP_N,intersections,dataSigmaBR,tran):
    error_m = np.linalg.norm(r_LP_N - intersections, axis=1)
    plt.plot(timeLineSet[120:],error_m[120:])
    vectorData = dataSigmaBR
    sNorm = np.array([4*np.arctan(np.linalg.norm(vectorData[i]))*(np.linalg.norm(tran[i])-astroFunctions.E_radius * 1e3) for i in range(len(vectorData))])
    plt.plot(timeLineSet[120:], sNorm[120:],'--',
             color=unitTestSupport.getLineColor(1, 3),
             )
    plt.xlabel('Time [min]')
    plt.ylabel('Error [m]')
    plt.legend(['Error', 'Approximated Error using θ'], loc='upper center')
    plt.show()
    


#Running function 
def run(show_plots):
    """
    The scenarios can be run with the followings setups parameters:

    Args:
        show_plots (bool): Determines if the script should display plots

    """

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    # set the simulation time variable used later on
    simulationTime = macros.min2nano(6.0)

    #
    #  create the simulation process
    #
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(1.0)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    #
    #   setup the simulation tasks/objects
    #

    # initialize spacecraft object and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"
    # define the simulation inertia
    I = [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0]
    scObject.hub.mHub = 750.0  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [
        [0.0],
        [0.0],
        [0.0],
    ]  
    # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    # add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)

    # clear prior gravitational body and SPICE setup definitions
    gravFactory = simIncludeGravBody.gravBodyFactory()

    # setup Earth Gravity Body
    earth = gravFactory.createEarth()
    earth.isCentralBody = True  # ensure this is the central gravitational body
    mu = earth.mu

    # attach gravity model to spacecraft
    gravFactory.addBodiesTo(scObject)

    #
    #   initialize Spacecraft States with initialization variables
    #
    # setup the orbit using classical orbit elements
    oe = orbitalMotion.ClassicElements()
    oe.a = (6378 + 600) * 1000.0  # meters
    oe.e = 0.1
    oe.i = 63.3 * macros.D2R
    oe.Omega = 88.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 125.3 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN  # m   - r_CN_N
    scObject.hub.v_CN_NInit = vN  # m/s - v_CN_N
    scObject.hub.sigma_BNInit = [[0.1], [0.2], [-0.3]]  # sigma_BN_B
    scObject.hub.omega_BN_BInit = [[0.001], [-0.01], [0.03]]  # rad/s - omega_BN_B

    # setup extForceTorque module
    # the control torque is read in through the messaging system
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "externalDisturbance"
    # use the input flag to determine which external torque should be applied
    # Note that all variables are initialized to zero.  Thus, not setting this
    # vector would leave it's components all zero for the simulation.
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask(simTaskName, extFTObject)

    # add the simple Navigation sensor module.  This sets the SC attitude, rate, position
    # velocity navigation message
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject)
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    # Create Strip Boulder Santiago
    striptarget = stripLocation.StripLocation()
    striptarget.ModelTag = "ImagingBoulderDenverStrip"
    striptarget.planetRadius = astroFunctions.E_radius * 1e3
    striptarget.acquisition_speed = 3*1e-6
    striptarget.specifyLocationStart(np.radians(39.99), np.radians(-105.26), 0)
    striptarget.specifyLocationEnd(np.radians(39.99), np.radians(-105.26), 0)
    striptarget.lenght_line()
    # print(striptarget.theta)
    # print(striptarget.lenght_central_line)
    # print(striptarget.p_start)
    # print(striptarget.p_end)
    # print(striptarget.r_LP_P_Start)
    # print(striptarget.r_LP_P_End)


    # striptarget.specifyLocationStart(np.radians(39.74), np.radians(-104.99), 1607)
    # striptarget.specifyLocationEnd(np.radians(39.74), np.radians(-104.99), 1607)
    striptarget.minimumElevation = np.radians(10.0)
    striptarget.maximumRange = 1e9
    striptarget.addSpacecraftToModel(scObject.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, striptarget)

   

    #
    #   setup the FSW algorithm tasks
    #


    # setup Boulder pointing guidance module
    locPoint = locationPointing.locationPointing()
    locPoint.ModelTag = "locPoint"
    scSim.AddModelToTask(simTaskName, locPoint)
    locPoint.pHat_B = [0, 0, 1]
    locPoint.scAttInMsg.subscribeTo(sNavObject.attOutMsg)
    locPoint.useBoresightRateDamping = 1
    locPoint.scTransInMsg.subscribeTo(sNavObject.transOutMsg)
    locPoint.locationInMsg.subscribeTo(striptarget.currentGroundStateOutMsg)

    # setup the MRP Feedback control module
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    scSim.AddModelToTask(simTaskName, mrpControl)
    mrpControl.guidInMsg.subscribeTo(locPoint.attGuidOutMsg)
    mrpControl.K = 16
    mrpControl.Ki = -1 # make value negative to turn off integral feedback
    mrpControl.P = 30.0
    mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1
 

    # connect torque command to external torque effector
    extFTObject.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)

    #
    #   Setup data logging before the simulation is initialized
    #
    #numDataPoints = 100
    #samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)
    samplingTime=3000000000
    mrpLog = mrpControl.cmdTorqueOutMsg.recorder(samplingTime)
    attErrLog = locPoint.attGuidOutMsg.recorder(samplingTime)
    snAttLog = sNavObject.attOutMsg.recorder(samplingTime)
    snTransLog = sNavObject.transOutMsg.recorder(samplingTime)
    locationLog = striptarget.accessOutMsgs[-1].recorder(samplingTime)
    target = striptarget.currentGroundStateOutMsg.recorder(samplingTime)
    
    scSim.AddModelToTask(simTaskName, mrpLog)
    scSim.AddModelToTask(simTaskName, attErrLog)
    scSim.AddModelToTask(simTaskName, snAttLog)
    scSim.AddModelToTask(simTaskName, snTransLog)
    scSim.AddModelToTask(simTaskName, locationLog)
    scSim.AddModelToTask(simTaskName, target)

    #
    # create simulation messages
    #

    # create the FSW vehicle configuration message
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = I  # use the same inertia in the FSW algorithm as in the simulation
    configDataMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
    mrpControl.vehConfigInMsg.subscribeTo(configDataMsg)

     # if this scenario is to interface with the BSK Viz, uncomment the following lines
    if vizSupport.vizFound:
        viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject
                                                  , saveFile=fileName
                                                  )
        
        vizSupport.addLocation(viz, stationName="Start Strip"
                               , parentBodyName=earth.displayName
                               , r_GP_P=unitTestSupport.EigenVector3d2list(striptarget.r_LP_P_Start)
                               , fieldOfView=np.radians(160.)
                               , color='pink'
                               , range=2000.0*1000  # meters
                               )

        vizSupport.addLocation(viz,stationName="End Strip",
            parentBodyName=earth.displayName,
            r_GP_P=[-1999589.8519127548,-4929768.147176737,3518472.552200294],
            fieldOfView=np.radians(160.0),
            color="green",
            range=2000.0 * 1000,  # meters
            )
        viz.settings.spacecraftSizeMultiplier = 1.5
        viz.settings.showLocationCommLines = 1
        viz.settings.showLocationCones = 1
        viz.settings.showLocationLabels = 1

    #
    #   initialize Simulation
    #
    scSim.InitializeSimulation()

    #
    #   Run the simulation to point the camera towards Boulder 
    #
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    # #
    # #   Update the location of the end of the strip to start imaging the strip
    # #
    striptarget.specifyLocationEnd(np.radians(33.48), np.radians(-112.078232), 0)
    #striptarget.specifyLocationEnd(np.radians(39.99), np.radians(-105.26), 0)
    

    # # #
    # # #   configure the new simulation stop time and execute sim
    # # #
    simulationTime2 = macros.min2nano(11.0)
    scSim.ConfigureStopTime(simulationTime + simulationTime2)
    scSim.ExecuteSimulation()

    #
    #   retrieve the logged data
    #
    dataLr = mrpLog.torqueRequestBody
    dataSigmaBR = attErrLog.sigma_BR
    dataOmegaBR = attErrLog.omega_BR_B
    hasAccess = locationLog.hasAccess
    r_LP_N = target.r_LP_N
    att=snAttLog.sigma_BN
    tran=snTransLog.r_BN_N
    np.set_printoptions(precision=16)
 

    if show_plots:
    #
    #   plot the results
    #
        timeLineSet = attErrLog.times() * macros.NANO2MIN
        plt.close("all")  # clears out plots from earlier test runs

        plot_attitude_error(timeLineSet, dataSigmaBR, simulationTime*macros.NANO2MIN)
        figureList = {}
        pltName = fileName + "1"
        figureList[pltName] = plt.figure(1)

        plot_control_torque(timeLineSet, dataLr, simulationTime*macros.NANO2MIN)
        pltName = fileName + "2"
        figureList[pltName] = plt.figure(2)

        plot_rate_error(timeLineSet, dataOmegaBR, simulationTime*macros.NANO2MIN)

        plot_access(timeLineSet, hasAccess, simulationTime*macros.NANO2MIN)

        camera_vectors = np.array([transform_body_to_inertial(att[i], locPoint.pHat_B) for i in range(len(att))])
        intersections=np.array([ray_sphere_intersection(tran[i], transform_body_to_inertial(att[i], locPoint.pHat_B), astroFunctions.E_radius * 1e3) for i in range(len(tran))])
        
        # Plot the full trajectory in a separate figure
        plot_full_trajectory(r_LP_N, astroFunctions.E_radius * 1e3,tran, camera_vectors,intersections)
        plot_error_m(timeLineSet,r_LP_N,intersections,dataSigmaBR,tran)

        

        #print(r_LP_N

        #plot_intersection(tran,locPoint.pHat_B,astroFunctions.E_radius * 1e3,att)

        # plot_3D_R_LP_N(timeLineSet, r_LP_N, astroFunctions.E_radius * 1e3)
        #generate_video(timeLineSet, r_LP_N, astroFunctions.E_radius * 1e3, plot_earth=True, zoom_trajectory=True)

  
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    return figureList




#
# This statement below ensures that the unit test scrip can be run as a
# stand-along python script
#
if __name__ == "__main__":
    run(
        True  # show_plots
    )

