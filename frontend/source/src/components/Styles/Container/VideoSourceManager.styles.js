import styled from 'styled-components';

export const ManagerContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  width: 100%;
  max-height: 40vh;         
  overflow-y: auto;         
  overflow-x: hidden;
  box-sizing: border-box;
  padding-right: 10px;
  margin: 5px;
`;

export const SubmitButtonWrapper = styled.div`
  display: flex;
  justify-content: flex-end;
  width: 100%;
  margin-top: 1rem;
`;

export const SubmitButton = styled.button`
  padding: 10px 20px;
  background-color: #007bff;
  border: none;
  border-radius: 6px;
  color: white;
  font-weight: bold;
  cursor: pointer;

  &:hover {
    background-color: #0056b3;
  }
`;
