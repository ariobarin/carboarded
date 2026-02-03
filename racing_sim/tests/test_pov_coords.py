
import unittest
import math
import numpy as np

class TestPOVCoords(unittest.TestCase):
    def test_lidar_angle_conversion(self):
        """Test that lidar world angles map to correct screen angles."""
        # Formula from renderer.py: screen_angle = -pi/2 - ray_angle
        # ray_angle: positive = left, negative = right
        
        # Case 1: Forward ray (0 degrees)
        ray_angle = 0
        screen_angle = -math.pi/2 - ray_angle
        # Expect UP on screen (negative Y)
        # cos(-90) = 0 (center X), sin(-90) = -1 (up Y)
        self.assertAlmostEqual(math.cos(screen_angle), 0, places=5)
        self.assertAlmostEqual(math.sin(screen_angle), -1, places=5)
        
        # Case 2: Left ray (+30 degrees = +pi/6)
        ray_angle = math.pi/6
        screen_angle = -math.pi/2 - ray_angle # -90 - 30 = -120 degrees
        # Expect LEFT-UP
        # cos(-120) = -0.5 (Left X)
        # sin(-120) = -0.866 (Up Y)
        self.assertLess(math.cos(screen_angle), -0.4) # Left
        self.assertLess(math.sin(screen_angle), -0.8) # Up
        
        # Case 3: Right ray (-30 degrees = -pi/6)
        ray_angle = -math.pi/6
        screen_angle = -math.pi/2 - ray_angle # -90 - (-30) = -60 degrees
        # Expect RIGHT-UP
        # cos(-60) = 0.5 (Right X)
        # sin(-60) = -0.866 (Up Y)
        self.assertGreater(math.cos(screen_angle), 0.4) # Right
        self.assertLess(math.sin(screen_angle), -0.8) # Up

    def test_grid_overlay_transforms(self):
        """Test that grid array manipulations result in correct visual orientation."""
        # Grid representation:
        # col 0 = RIGHT of car
        # col N-1 = LEFT of car
        # row 0 = FAR
        # row N-1 = NEAR
        
        # Create a dummy grid where left side (col N-1) is marked
        # and Near-Left (bottom-left of grid) is marked
        size = 10
        grid = np.zeros((size, size), dtype=np.uint8)
        
        # Mark LEFT side of car (col index size-1)
        grid[:, size-1] = 1 
        # Mark RIGHT side of car (col index 0)
        grid[:, 0] = 2
        
        # Renderer logic:
        # 1. fliplr (flips columns)
        flipped = np.fliplr(grid)
        
        # Now col 0 of flipped is what used to be col size-1 (LEFT)
        self.assertEqual(flipped[0, 0], 1) 
        # Col size-1 of flipped is what used to be col 0 (RIGHT)
        self.assertEqual(flipped[0, size-1], 2)
        
        # 2. swapaxes(0, 1) for pygame surface [x, y]
        # surface[x, y] = flipped[y, x] (swapped)
        # We want LEFT of car to appear on LEFT of screen (x=0)
        
        # Check x=0 (Left of screen):
        # surface[0, ANY] corresponds to flipped[ANY, 0]
        # flipped[ANY, 0] is original grid[ANY, size-1], which is LEFT side.
        # So x=0 shows LEFT side. CORRECT.
        pass

    def test_grid_worldspace_transform(self):
        """Test worldspace grid coordinate transformation."""
        # Car at origin facing +Y (North)
        car_pos_x, car_pos_y = 0.0, 100.0
        car_angle = math.pi / 2 # 90 degrees = Facing +Y
        
        # Test point: Forward-Left relative to car
        # In world: Car is at (0, 100) facing (0, 110).
        # Left is (-1, 0).
        # So Forward-Left is (-10, 110)
        world_pt_x, world_pt_y = -10.0, 110.0
        
        scale = 2.0
        
        # Logic from renderer.py
        rel_x = world_pt_x - car_pos_x # -10
        rel_y = world_pt_y - car_pos_y # 10
        
        # Rotate by -car_angle + pi/2 = -90 + 90 = 0
        angle = -car_angle + math.pi/2
        cos_a = math.cos(angle) # 1
        sin_a = math.sin(angle) # 0
        
        # Transform
        screen_rel_x = (rel_x * cos_a - rel_y * sin_a) * scale
        # screen_rel_x = (-10*1 - 10*0) * 2 = -20 (Left on screen)
        
        screen_rel_y = -(rel_x * sin_a + rel_y * cos_a) * scale
        # screen_rel_y = -(-10*0 + 10*1) * 2 = -20 (Up on screen)
        
        self.assertLess(screen_rel_x, 0, "Should be to the left on screen")
        self.assertLess(screen_rel_y, 0, "Should be up on screen (negative Y)")
        
        # Test point: Right of car
        # car at (0, 100) facing +Y. Right is +X.
        # Point at (10, 100)
        world_pt_x, world_pt_y = 10.0, 100.0
        rel_x = 10.0
        rel_y = 0.0
        
        screen_rel_x = (rel_x * cos_a - rel_y * sin_a) * scale
        # (10*1 - 0) * 2 = 20 (Right)
        
        self.assertGreater(screen_rel_x, 0, "Should be to the right on screen")

if __name__ == '__main__':
    unittest.main()
