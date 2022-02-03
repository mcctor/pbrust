use std::ops::{Add, AddAssign, Index, IndexMut, Mul};

pub type Vector2i = Vector2<i32>;
pub type Vector2f = Vector2<f32>;
pub type Vector3i = Vector3<i32>;
pub type Vector3f = Vector3<f32>;

trait Vector<T>: Index<usize> + IndexMut<usize> {}

#[derive(Copy, Clone)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vector2<T> {
    pub fn new(x: T, y: T) -> Self {
        Vector2 { x, y }
    }
}

impl<T> Vector<T> for Vector2<T> {}

impl<T: AddAssign + Add<Output=T>> Add<Vector2<T>> for Vector2<T> {
    type Output = Vector2<T>;

    fn add(self, rhs: Vector2<T>) -> Self::Output {
        Vector2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: AddAssign + Add<Output=T>> AddAssign<Vector2<T>> for Vector2<T> {
    fn add_assign(&mut self, rhs: Vector2<T>) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: Mul<Output=T> + Copy + Clone> Mul<T> for Vector2<T> {
    type Output = Vector2<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Vector2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Mul<Vector2<i32>> for i32 {
    type Output = Vector2<i32>;

    fn mul(self, rhs: Vector2<i32>) -> Self::Output {
        rhs * self
    }
}

impl Mul<Vector2<f32>> for f32 {
    type Output = Vector2<f32>;

    fn mul(self, rhs: Vector2<f32>) -> Self::Output {
        rhs * self
    }
}

impl<T> Index<usize> for Vector2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index provided non-existent"),
        }
    }
}

impl<T> IndexMut<usize> for Vector2<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index provided non-existent"),
        }
    }
}

#[derive(Copy, Clone)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vector3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector3 { x, y, z }
    }
}

impl<T> Vector<T> for Vector3<T> {}

impl<T: AddAssign + Add<Output=T>> Add<Vector3<T>> for Vector3<T> {
    type Output = Vector3<T>;

    fn add(self, rhs: Vector3<T>) -> Self::Output {
        Vector3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: AddAssign + Add<Output=T>> AddAssign<Vector3<T>> for Vector3<T> {
    fn add_assign(&mut self, rhs: Vector3<T>) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl<T: Mul<Output=T> + Copy + Clone> Mul<T> for Vector3<T> {
    type Output = Vector3<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Vector3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vector3<i32>> for i32 {
    type Output = Vector3<i32>;

    fn mul(self, rhs: Vector3<i32>) -> Self::Output {
        rhs * self
    }
}

impl Mul<Vector3<f32>> for f32 {
    type Output = Vector3<f32>;

    fn mul(self, rhs: Vector3<f32>) -> Self::Output {
        rhs * self
    }
}

impl<T> Index<usize> for Vector3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index provided non-existent"),
        }
    }
}

impl<T> IndexMut<usize> for Vector3<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index provided non-existent")
        }
    }
}

#[cfg(test)]
mod test_vector_ops {
    use super::{Vector2, Vector2f, Vector2i, Vector3};

    #[test]
    fn indexing() {
        let vec2 = Vector2 { x: 2, y: 3 };
        assert_eq!(vec2[0], 2);
        assert_eq!(vec2[1], 3);

        let vec3 = Vector3 { x: 4, y: 5, z: 6 };
        assert_eq!(vec3[0], 4);
        assert_eq!(vec3[1], 5);
        assert_eq!(vec3[2], 6);
    }

    #[test]
    fn creation() {
        let vec2 = Vector2::new(1, 2);
        assert_eq!(vec2.x, 1);
        assert_eq!(vec2.y, 2);

        let vec3 = Vector3::new(1, 2, 3);
        assert_eq!(vec3.x, 1);
        assert_eq!(vec3.y, 2);
        assert_eq!(vec3.z, 3);
    }

    #[test]
    fn addition_op() {
        let vec1 = Vector2::new(0, 2);
        let vec2 = Vector2::new(1, 3);
        let res1 = vec1 + vec2;
        assert_eq!(res1.x, 1);
        assert_eq!(res1.y, 5);

        let vec3 = Vector3::new(0, 2, 0);
        let vec4 = Vector3::new(1, 3, 1);
        let res2 = vec3 + vec4;
        assert_eq!(res2.x, 1);
        assert_eq!(res2.y, 5);
        assert_eq!(res2.z, 1);
    }

    #[test]
    fn scalar_mul() {
        let vec1 = Vector2::new(1, 1);
        let res = vec1 * 5;
        assert_eq!(res.x, 5);
        assert_eq!(res.y, 5);

        let vec2 = Vector3::new(3, 3, 3);
        let res2 = vec2 * 5;
        assert_eq!(res2.x, 15);
        assert_eq!(res2.y, 15);
        assert_eq!(res2.z, 15);

        // commutative test
        let vec3 = Vector3::new(3, 3, 3);
        let res3 = 5 * vec3;
        assert_eq!(res3.x, 15);
        assert_eq!(res3.y, 15);
        assert_eq!(res3.z, 15);

        let vec4 = Vector2::new(2, 2);
        let res4 = 4 * vec4;
        assert_eq!(res4.x, 8);
        assert_eq!(res4.y, 8);
    }

    #[test]
    fn scalar_div() {
        let vec1 = Vector2::new(25, 25);
        let res = vec1 / 5;
        assert_eq!(res.x, 5);
        assert_eq!(res.y, 5);

        let vec2 = Vector3::new(30, 30, 30);
        let res2 = vec2 / 2;
        assert_eq!(res2.x, 15);
        assert_eq!(res2.y, 15);
        assert_eq!(res2.z, 15);
    }
}