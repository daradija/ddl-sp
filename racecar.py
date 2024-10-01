import pygame
import sys
import random
import math
import numpy as np

# Inicializa pygame
pygame.init()

# Configura las dimensiones de la ventana
screen_width = 800*2
screen_height = 500*2
screen = pygame.display.set_mode((screen_width, screen_height))

# Colores
black = (0, 0, 0)

class Regresor:
	def __init__(self):
		self.hasta=1000
		self.desde=100
		self.xs=[]
		self.ys=[]
		self.x0=None

		self.B=None

	def similar(self,a,b):
		return min(a,b)#/max(a,b)

	def predict(self,xs):
		profundidad=3
		keys=[[0,0,1],
				[0,1,1],
				# [1,0,0],
				# [0,1,0],
				[1,0,1]]
		base=[(None,np.array(xs))]
		mejorHeuristico=-math.inf
		r=None
		for prof in range(profundidad):
			base2=[]
			for k,b in base:
				for key in keys:
					xs2=np.concatenate((b,key))
					ys=np.dot(xs2.T,self.B)
					if prof==profundidad-1:
						#heuristico=self.similar(ys[0],ys[4])*self.similar(ys[1],ys[3])*ys[2]
						heuristico=self.similar(ys[0],ys[6])*self.similar(ys[1],ys[5])*self.similar(ys[2],ys[4])*ys[3]
						#heuristico=sum(ys)
						if heuristico>mejorHeuristico:
							mejorHeuristico=heuristico
							r=k
					else:
						if prof==0:
							base2.append((key,ys))
						else:
							base2.append((k,ys))
			#keys=[(0,0,1)]
			base=base2
		return (True if r2==1 else False for r2 in r)

	def add(self,xs,vel,leftb,rightb,upb):
		if not self.x0:
			self.x0=xs
			return 
		left=1 if leftb else 0
		right=1 if rightb else 0
		up=1 if upb else 0
		self.xs.append(self.x0+[vel,left,right,up])
		self.ys.append(xs+[vel])
		if len(self.xs)>self.hasta:
			X=np.array(self.xs)
			Y=np.array(self.ys)
			self.B=np.linalg.lstsq(X,Y,rcond=None)[0]
			xs2=[]
			ys2=[]
			for i in [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]:
				num=0
				for j in range(len(self.xs)-1,-1,-1):
					xs3=self.xs[j]
					tu=tuple(xs3[-3:])
					if tu==i:
						num+=1
						xs2.append(xs3)
						ys3=self.ys[j]
						ys2.append(ys3)
					if self.desde/8<=num:
						break
			self.xs=xs2
			self.ys=ys2

 
		self.x0=xs

class Pulsante:
	def __init__(self):
		self.x0=None
		self.t=0

	def predict(self,xs):
		return 

	def add(self,xs,vel,leftb,rightb,upb):
		if not self.x0:
			self.x0=xs
			return 
		left=1 if leftb else 0
		right=1 if rightb else 0
		up=1 if upb else 0
		xt=self.x0+[vel,left,right,up]
		xt=[int(x+1) for x in xt]
		yt=xs+[vel]
		yt=[int(y+1) for y in yt]


		ny=[[] for _ in range(len(yt))]

		for iy in range(len(yt)):
			x=[0]*len(xt)
			for t in range(1,1000):
				for i in range(len(x)):
					if t%xt[i]==0:
						x[i]+=1
				if t%yt[iy]==0:
					ny[iy].append(x)
					x=[0]*len(xt)
			

		print(ny)
		# inicial
		# tiempo transcurrido desde el último disparo en t
		

class Car:
	def __init__(self,color=(255,0,0),position=[screen_width//2,screen_height//2],radius=50,velocity=[2,2],image=None,rozamiento=1.0):
		self.color = color
		self.position = position.copy()
		self.radius = radius
		self.velocity = velocity.copy()
		self.motor=[0,0]
		self.image = image
		self.rozamiento = rozamiento
		self.angle=math.pi
		self.left=False
		self.right=False
		self.giroVel=1/180
		self.maxVelocity=0.005
		self.nn=Regresor()
		#self.nn=Pulsante()

	def drawHand(self,left,right,up,learn,*circuit):
		q0=self.position
		view=[50,100,200,400,200,100,50]
		incAngulo=[-math.pi/2,-math.pi/4,-math.pi/8,0,math.pi/8,math.pi/4,math.pi/2]
		xs=[]
		for k,inc in enumerate(incAngulo):
			v=view[k]
			angle=self.angle+inc
			q1=(self.position[0]+math.sin(-angle)*v,self.position[1]+math.cos(-angle)*v)
			pygame.draw.line(screen, (0,0,255), q0, q1, 5)
			min=distancia(resta(q0,q1))
			for c in circuit:
				for i in range(len(c.points)):
					p0=c.points[i]
					p1=c.points[(i+1)%len(c.points)]

					i=interseccion(q0,q1,p0,p1)
					if i:
						#print(cand)
						cand=distancia(resta(q0,i))
						if cand<min:
							pygame.draw.circle(screen, (255,0,0), (int(i[0]),int(i[1])),10)
							min=cand
			xs.append(min)
		vel=distancia(self.velocity)
		if learn:
			self.nn.add(xs,vel,left,right,up)
		return xs+[vel]

	def pensarYActuar(self):
		self.moveUp(self.maxVelocity)
		self.aDondeIr()

		if self.left:
			self.turnLeft()

		if self.right:
			self.turnRight()

	def turnRight(self):
		self.angle+=math.pi*self.giroVel
		self.moveUp()
		#ball.velocity[0] += 1

	def turnLeft(self):
		self.angle-=math.pi*self.giroVel
		self.moveUp()
		#ball.velocity[0] -= 1

	def down(self):
		self.motor=multiplicarEscalar(0.9,self.motor)

	def aDondeIr(self):
		ball=self
		if not hasattr(ball,"ext") or not hasattr(ball,"int"):
			return
		target=multiplicarEscalar(0.5,suma(ball.ext,ball.int))
		
		vdir=resta(target,ball.position)
		#if distancia(vdir)>10:
		a=-math.atan2(vdir[0],vdir[1])
		while a<0:
			a+=2*math.pi
		while a>2*math.pi:
			a-=2*math.pi
		while ball.angle<0:
			ball.angle+=2*math.pi
		while ball.angle>2*math.pi:
			ball.angle-=2*math.pi

		dif=abs(a-ball.angle)
		#if dif>math.pi/10:
		self.left=False
		self.right=False
		if dif<math.pi:
			if a>ball.angle:
				self.right=True
			if a<ball.angle:
				self.left=True
		else:
			if a>ball.angle:
				self.left=True
			if a<ball.angle:
				self.right=True
		#ball.angle=a

	def moveUp(self,vel=0.0):
		vel=distancia(self.motor)+vel
		self.motor[0] = math.sin(-self.angle)*vel
		self.motor[1] = math.cos(-self.angle)*vel

	def update(self):
		# vel=distancia(self.velocity)
		# self.velocity[0] = math.sin(-self.angle)*vel
		# self.velocity[1] = math.cos(-self.angle)*vel

		self.position[0] += self.velocity[0]+self.motor[0]
		self.position[1] += self.velocity[1]+self.motor[1]

		if self.position[0] <= -self.radius:
			self.position[0] += screen_width+2*self.radius
		if screen_width+self.radius <= self.position[0]:
			self.position[0] -= screen_width+2*self.radius

		if self.position[1] <= self.radius or self.position[1] >= screen_height - self.radius:
			self.velocity[1] = -self.velocity[1]

		if self.position[1] <= self.radius:
			self.position[1] = self.radius

		if screen_height - self.radius <= self.position[1]:
			self.position[1] = screen_height - self.radius

		# 1 no puedo tomar curvas porque derrapa mucho, no dobla el volante
		# 0 atravieso las paredes
		trasferencia=0.05
		
		self.velocity[0] = self.velocity[0]*self.rozamiento+self.motor[0]*trasferencia
		self.velocity[1] = self.velocity[1]*self.rozamiento+self.motor[1]*trasferencia
		self.motor[0]=self.motor[0]*(1-trasferencia)
		self.motor[1]=self.motor[1]*(1-trasferencia)

	def draw(self, pygame):
		# if self.image:
		# 	image = pygame.image.load(self.image)
		# 	image = pygame.transform.scale(image, (self.radius*2, self.radius*2))
		# 	screen.blit(image, (self.position[0]-self.radius, self.position[1]-self.radius))
		# else:
		# 	pygame.draw.circle(screen, self.color, self.position, self.radius)

		proporcion=math.pi/6
		d=(self.radius,0)
		d0=girar(d, math.pi/2-proporcion+self.angle)
		d05=girar(d, math.pi/2+self.angle)
		d1=girar(d, math.pi/2+proporcion+self.angle)
		d2=girar(d, math.pi*3/2-proporcion+self.angle)
		d3=girar(d, math.pi*3/2+proporcion+self.angle)
		d0=suma(d0,self.position)
		d05=suma(d05,self.position)
		d1=suma(d1,self.position)
		d2=suma(d2,self.position)
		d3=suma(d3,self.position)
		pygame.draw.polygon(screen, self.color, [d0,d05,d1,d2,d3])
		# pygame.draw.line(screen, self.color, d0, d1)
		# pygame.draw.line(screen, self.color, d1, d2)
		# pygame.draw.line(screen, self.color, d2, d3)
		# pygame.draw.line(screen, self.color, d3, d0)
		
	def resetEnergy(self):
		self.energy = [0,0]

	def transferEnergy(self,energy,p):
		auxX= energy*(p.velocity[0]*p.radius)
		self.energy[0] +=auxX
		p.energy[0] -=auxX
		auxY= energy*(p.velocity[1]*p.radius)
		self.energy[1] +=auxY
		p.energy[1] -=auxY

	def applyEnergy(self):
		self.velocity[0] += self.energy[0]/self.radius
		self.velocity[1] += self.energy[1]/self.radius


class Circuit:
	def __init__(self,ext=None):
		self.points = []
		self.ext=ext

	def add(self, point):
		self.points.append(point)

		print(self.points)

		if self.ext:
			if 2<len(self.points):
				ps1=[]
				
				for i in range(len(self.points)):
					p0=self.points[i-1]
					p1=self.points[i]

					d=vnormalTam(p0,p1,100)
					d2=girar(d,-math.pi/2)
					d3=suma(d2,p0)
					ps1.append(d3)

					d=vnormalTam(p1,p0,100)
					d2=girar(d,math.pi/2)
					d3=suma(d2,p1)
					ps1.append(d3)

				self.ext.points=[]
				for i in range(0,len(ps1),2):
					p0=ps1[i-2]
					p1=ps1[i-1]
					p2=ps1[i]
					p3=ps1[(i+1)%len(ps1)]

					pi=find_intersection(p0,p1,p2,p3)
					if pi:
						self.ext.points.append(pi)


	
	def draw(self, pygame):
		if len(self.points)<=1:
			return
		for i in range(len(self.points)-1):
			pygame.draw.line(screen, (0,255,0), self.points[i], self.points[i+1], 5)		
		pygame.draw.line(screen, (0,255,0), self.points[len(self.points)-1], self.points[0], 5)

	def collision(self,pygame,ps):
		if len(self.points)<=3:
			return
		
		for p in ps:
			cercano=math.inf
			memoria=None

			for i in range(len(self.points)):
				s0=self.points[i]
				if i==len(self.points)-1:
					s1=self.points[0]
				else:
					s1=self.points[i+1]
			

				d1=resta(s1,s0)
				d2=resta(p.position,s0)
				t3=productoEscalar(d1,d2)/distancia(d1)

				hasta=distancia(d1)
				valido=0<=t3 and t3<=hasta
				cand=0
				if valido:
					d4=multiplicarEscalar(t3,vectorNormalizado(d1))
					cand=distancia(resta(d2,d4))
					d5=suma(s0,d4)
				else:
					valido=True
					ds0=distancia(resta(p.position,s0))
					ds1=distancia(resta(p.position,s1))
					cand=min(ds0,ds1)
					if ds0<ds1:
						d5=s0
					else:
						d5=s1

				if valido:
					if cand<cercano:
						zpv=zProductoVectorial(d1,d2)
						memoria=(d5,zpv,s1)
						cercano=cand
			
			if memoria:
				d5,zpv,s1=memoria

				if not self.ext: # Es exterior
					zpv=-zpv
					p.ext=s1
				else:
					p.int=s1

				if zpv>0:
					#pygame.draw.circle(screen, (255,0,0), d5,10)
					pass
				else:
					direccionEmpuje=resta(p.position,d5)
					empuje=p.radius-distancia(direccionEmpuje)
					if empuje>0:
						den=vectorNormalizado(direccionEmpuje)
						vempuje=multiplicarEscalar(empuje,den)

						inicial=distancia(p.velocity)

						p.velocity[0] += vempuje[0]
						p.velocity[1] += vempuje[1]

						p.velocity=vnormalTam((0,0),p.velocity,inicial)

					#pygame.draw.circle(screen, (0,255,0), d5,10)


def suma(v1,v2):
	return [v1[0]+v2[0],v1[1]+v2[1]]

def multiplicarEscalar(a,v):
	return [a*v[0],a*v[1]]

def resta(v1,v2):
	return [v1[0]-v2[0],v1[1]-v2[1]]

def productoEscalar(v1,v2):
	return v1[0]*v2[0]+v1[1]*v2[1]

def distancia(v):
	return (v[0]**2+v[1]**2)**0.5

def vectorNormalizado(v):
	d=distancia(v)
	return [v[0]/d,v[1]/d]

def zProductoVectorial(v1,v2):
	return v1[0]*v2[1]-v1[1]*v2[0]

def vnormalTam(p0,p1,tam):
	d=resta(p1,p0)
	mul=tam/distancia(d)
	return [d[0]*mul,d[1]*mul]

def girar(v,angulo):
	return [v[0]*math.cos(angulo)-v[1]*math.sin(angulo),v[0]*math.sin(angulo)+v[1]*math.cos(angulo)]

def interseccion(p0,p1,p2,p3):
	# Verifica si el rectángulo r1 se solapa con el rectángulo r2
	# AABB: Axis-Aligned Bounding Box
	maxAx = max(p0[0], p1[0])
	minAx = min(p0[0], p1[0])
	maxBx = max(p2[0], p3[0])
	minBx = min(p2[0], p3[0])
	if maxAx < minBx or maxBx < minAx:
		return None
	maxAy = max(p0[1], p1[1])
	minAy = min(p0[1], p1[1])
	maxBy = max(p2[1], p3[1])
	minBy = min(p2[1], p3[1])
	if maxAy < minBy or maxBy < minAy:
		return None
	i=find_intersection(p0, p1, p2, p3)
	# is i inside p2p3?
	if min(p2[0],p3[0])<=i[0]<=max(p2[0],p3[0]) and min(p2[1],p3[1])<=i[1]<=max(p2[1],p3[1]):
		if min(p0[0],p1[0])<=i[0]<=max(p0[0],p1[0]) and min(p0[1],p1[1])<=i[1]<=max(p0[1],p1[1]):
			return i
	return None

def find_intersection(p0, p1, p2, p3):
    # Calcula la pendiente de las rectas r1 y r2
    m1 = (p1[1] - p0[1]) / (p1[0] - p0[0]) if p1[0] != p0[0] else float('inf')
    m2 = (p3[1] - p2[1]) / (p3[0] - p2[0]) if p3[0] != p2[0] else float('inf')

    # Calcula el intercepto y de las rectas r1 y r2
    b1 = p0[1] - m1 * p0[0]
    b2 = p2[1] - m2 * p2[0]

    # Verifica si las rectas son paralelas
    if m1 == m2:
        return None  # Las rectas son paralelas o coincidentes, no hay intersección única

    # Calcula el punto de intersección
    if m1 == float('inf'):  # r1 es vertical
        x = p0[0]
        y = m2 * x + b2
    elif m2 == float('inf'):  # r2 es vertical
        x = p2[0]
        y = m1 * x + b1
    else:
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

    return (x, y)



ps=[]
ball=Car(color=(255,0,0),image="yo.png",rozamiento=0.995,radius=15)
ps.append(ball)
for i in range(5):
	ps.append(Car(rozamiento=0.99,color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)),position=[random.randint(0,screen_width),random.randint(0,screen_height)],radius=random.randint(10,20),velocity=[random.randint(-5,5),random.randint(-5,5)]))

circuitExt=Circuit()
circuit=Circuit(circuitExt)

for p in [(180, 551), (174, 483), (170, 432), (175, 380), (192, 296), (220, 242), (247, 218), (309, 219), (403, 196), (476, 159), (547, 168), (617, 194), (695, 238), (766, 234), (823, 192), (872, 155), (948, 131), (1037, 125), (1152, 143), (1226, 182), (1286, 240), (1339, 325), (1334, 387), (1259, 442), (1154, 384), (1052, 304), (900, 333), (821, 501), (847, 616), (934, 695), (1018, 714), (1148, 733), (1355, 758), (1395, 793), (1400, 821), (1382, 853), (1325, 881), (1012, 839), (809, 860), (684, 728), (711, 603), (733, 382), (632, 275), (355, 315), (296, 430), (415, 540), (424, 618), (421, 775), (332, 843), (183, 831), (142, 753)]:
	circuit.add(p)
# Bucle principal del juego
hand=None
running = True
while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		elif event.type == pygame.MOUSEBUTTONDOWN:
				circuit.add(event.pos)

	keys = pygame.key.get_pressed()

	screen.fill(black)

	# Mueve la pelota con las teclas de flecha
	# ball.velocity[0] /=1.001
	# ball.velocity[1] /=1.001

	ball.left=False
	ball.right=False
	ball.up=False

	if keys[pygame.K_SPACE]:
		ball.left,ball.right,ball.up=ball.nn.predict(hand)
	
	
	if keys[pygame.K_UP] or ball.up:
		ball.moveUp(ball.maxVelocity)
		ball.up=True
		#ball.aDondeIr()

	if keys[pygame.K_DOWN]:
		ball.down()		

	if keys[pygame.K_LEFT] or ball.left:
		ball.turnLeft()
		ball.left=True

		
	if keys[pygame.K_RIGHT] or ball.right:
		ball.turnRight()
		ball.right=True

	for p in ps:
		if p!=ball:
			p.pensarYActuar()
	
	for p in ps:
		p.update()

	for p1 in ps:
		p1.resetEnergy()

	for p1 in ps:
		for p2 in ps:
			if p1 != p2:
				if (p1.position[0]-p2.position[0])**2+(p1.position[1]-p2.position[1])**2 < (p1.radius+p2.radius)**2:
					p1.transferEnergy(1.0,p2)
					# p1.velocity[0] = -p1.velocity[0]
					# p1.velocity[1] = -p1.velocity[1]
	for p1 in ps:
		p1.applyEnergy()

	# Llena el fondo de la pantalla con negro

	circuit.draw(pygame)
	circuitExt.draw(pygame)
	for p in ps:
		p.draw(pygame)

	circuit.collision(pygame,ps)
	circuitExt.collision(pygame,ps)

	hand=ball.drawHand(ball.left,ball.right,ball.up,not keys[pygame.K_SPACE], circuit,circuitExt)

	# Actualiza la pantalla
	pygame.display.flip()

	# Controla la velocidad del bucle
	pygame.time.delay(2)

# Finaliza pygame y cierra la ventana
pygame.quit()
sys.exit()
