use std::{cmp, ops};
use std::ops::{Add, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub};

pub type Vector2f = Vector2<f32>;
pub type Vector2i = Vector2<i32>;
pub type Vector3f = Vector3<f32>;
pub type Vector3i = Vector3<i32>;

#[derive(Debug, Default, Clone, Copy)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

impl<T: Copy + cmp::PartialOrd + Default + Neg<Output=T>> Vector2<T> {
    pub fn from(x: T, y: T) -> Vector2<T> {
        Vector2 { x, y }
    }

    pub fn abs(x: &Vector2<T>) -> Vector2<T> {
        Vector2 { x: abs(x.x), y: abs(x.y) }
    }
}

impl<T: ops::Add<Output=T>> Add for Vector2<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: ops::Sub<Output=T>> Sub for Vector2<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T> Neg for Vector2<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        todo!()
    }
}

impl<T: ops::Mul<Output=T>> Mul for Vector2<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

impl<T> MulAssign for Vector2<T> {
    fn mul_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl<T: Copy + ops::Mul<Output=T>> Mul<T> for Vector2<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T> DivAssign for Vector2<T> {
    fn div_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl<T: Copy + ops::Div<Output=T> + cmp::PartialEq + Default> Div<T> for Vector2<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        assert!(rhs != T::default());
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<T> Index<usize> for Vector2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("valid index are 0, 1")
        }
    }
}

impl<T> IndexMut<usize> for Vector2<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("valid index are 0, 1")
        }
    }
}

#[cfg(test)]
mod test_vector2 {
    use crate::core::geometry::Vector2;

    #[test]
    fn indexing() {
        let some_vec = Vector2::from(0, 1);
        assert!(some_vec[0] == 0 && some_vec[1] == 1);
    }

    #[test]
    #[should_panic]
    fn index_out_of_range() {
        let some_vec = Vector2::from(0, 0);
        assert_eq!(some_vec[2], 0);
    }

    #[test]
    fn add() {
        let origin = Vector2::from(1, 2);
        let some_vec = Vector2::from(1, -1);
        let res_vec = origin + some_vec;
        assert!(res_vec[0] == 2 && res_vec[1] == 1);
    }

    #[test]
    fn sub() {
        let origin = Vector2::from(0, 2);
        let some_vec = Vector2::from(1, -1);
        let res_vec = origin - some_vec;
        assert!(res_vec[0] == -1 && res_vec[1] == 3)
    }

    #[test]
    fn div() {
        let origin = Vector2::from(4.0, 10.0);
        let res = origin / 2.0;
        assert!(res.x == 2.0 && res.y == 5.0)
      }

    #[test]
    fn scalar_mult() {
        let origin = Vector2::from(5, 9);
        let res = origin * 4;
        assert!(res.x == 20 && res.y == 36)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Copy + cmp::PartialOrd + Default + Neg<Output=T>> Vector3<T> {
    pub fn from(x: T, y: T, z: T) -> Vector3<T> {
        Vector3 { x, y, z }
    }

    pub fn abs(x: &Vector3<T>) -> Vector3<T> {
        Vector3 { x: abs(x.x), y: abs(x.y), z: abs(x.z) }
    }
}

impl<T: ops::Add<Output=T>> Add for Vector3<T> {
    type Output = Vector3<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: ops::Sub<Output=T>> Sub for Vector3<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T> Neg for Vector3<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        todo!()
    }
}

impl<T> MulAssign for Vector3<T> {
    fn mul_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl<T: ops::Mul<Output=T>> Mul for Vector3<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl<T: Copy + ops::Mul<Output=T>> Mul<T> for Vector3<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T> DivAssign for Vector3<T> {
    fn div_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl<T: Copy + ops::Div<Output=T> + cmp::PartialEq + Default> Div<T> for Vector3<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        assert!(rhs != T::default());
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T: Copy> Index<usize> for Vector3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("valid index are 0, 1, 2")
        }
    }
}

impl<T: Copy> IndexMut<usize> for Vector3<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("valid index are 0, 1, 2")
        }
    }
}

#[cfg(test)]
mod test_vector3 {
    use crate::core::geometry::Vector3;

    #[test]
    fn indexing() {
        let some_vec = Vector3::from(0, 1, 0);
        assert!(some_vec[0] == 0 && some_vec[1] == 1 && some_vec[2] == 0);
    }

    #[test]
    #[should_panic]
    fn index_out_of_range() {
        let some_vec = Vector3::from(0, 0, 3);
        assert_eq!(some_vec[3], 0);
    }

    #[test]
    fn add() {
        let origin = Vector3::from(1, 2, 1);
        let some_vec = Vector3::from(1, -1, 0);
        let res_vec = origin + some_vec;
        assert!(res_vec[0] == 2 && res_vec[1] == 1 && res_vec[2] == 1);
    }

    #[test]
    fn sub() {
        let origin = Vector3::from(0, 2, 2);
        let some_vec = Vector3::from(1, -1, -2);
        let res_vec = origin - some_vec;
        assert!(res_vec[0] == -1 && res_vec[1] == 3 && res_vec[2] == 4);
    }

    #[test]
    fn div() {
        let origin = Vector3::from(4.0, 10.0, 20.0);
        let res = origin / 2.0;
        assert!(res.x == 2.0 && res.y == 5.0 && res.z == 10.0);
    }

    #[test]
    fn scalar_mult() {
        let origin = Vector3::from(5, 9, 10);
        let res = origin * 4;
        assert!(res.x == 20 && res.y == 36 && res.z == 40);
    }
}

fn abs<T: cmp::PartialOrd + Default + Neg<Output = T>>(value: T) -> T {
    if value < T::default() {
        return -value
    }
    value
}